"""Durable, offline-first local stores for longitudinal Journey records."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import tempfile
import threading
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, TypeVar, cast

from openmed.clinical.journey_contracts import (
    ClinicalArtifact,
    ClinicalFact,
    ConflictSet,
    DatasetSnapshot,
    EvidenceLocator,
    ResolutionEvent,
    canonical_digest,
    sha256_digest,
)
from openmed.core.key_provider import EncryptionError

from .encryption import (
    ENCRYPTION_STATE_MIGRATING,
    PLAINTEXT_JOURNEY_CODEC,
    EncryptedJourneyCodec,
    JourneyEncryptionError,
    JourneyPayloadCodec,
    JourneyStoreEncryption,
    LocalJourneyReencryptionReport,
    PersistedEncryptionState,
    PlaintextJourneyCodec,
    ReencryptionReport,
    bind_encryption_state,
    encrypted_state,
    require_journey_encryption,
)
from .migrations import LATEST_MIGRATION_VERSION, MIGRATIONS
from .protocols import (
    AllowAllStoragePolicy,
    CanonicalRecord,
    CanonicalRecordVersion,
    CommitStatusUnknown,
    JobMetadata,
    StoragePolicy,
    StorePoint,
    StoreResult,
    StoreState,
    assert_contract_compatibility,
)

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_OPAQUE_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}_[A-Za-z0-9_-]{16,128}$")
_TIMESTAMP_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]+)?(?:Z|[+-][0-9]{2}:[0-9]{2})$"
)
_BLOB_NAME_RE = re.compile(r"^[0-9a-f]{64}$")
_BLOB_TEMP_PREFIX = ".journey-blob-"
_ENCRYPTION_MARKER = "encryption.json"
_ENCRYPTION_MARKER_SCHEMA_VERSION = 1
_MARKER_FIELDS = frozenset(
    {"envelope_version", "key_purpose", "key_verifiers", "schema_version", "state"}
)

# Primary-key columns bound into each payload envelope as associated data.
PAYLOAD_RECORD_KEYS: Mapping[str, tuple[str, ...]] = {
    "artifacts": ("artifact_id",),
    "evidence_locators": ("locator_id",),
    "clinical_facts": ("fact_id",),
    "conflict_sets": ("conflict_id",),
    "resolution_events": ("resolution_id",),
    "dataset_snapshots": ("snapshot_id",),
    "canonical_record_versions": ("canonical_id", "version"),
    "job_metadata_versions": ("job_id", "version"),
}

T = TypeVar("T")


class LocalStoreError(RuntimeError):
    """Base value-safe exception for local-store lifecycle failures."""


class StoreCompatibilityError(LocalStoreError):
    """Raised when persisted schema state is newer or has drifted."""


class StoreMigrationError(LocalStoreError):
    """Raised when a deterministic migration cannot be applied."""


class StoreConstraintError(LocalStoreError):
    """Raised by backend adapters for a value-safe constraint conflict."""


@dataclass(frozen=True, slots=True)
class IngestedGraph:
    """Value-safe counts for one committed source-to-fact graph."""

    artifact: ClinicalArtifact = field(repr=False)
    evidence_count: int
    fact_count: int
    conflict_count: int


class LocalArtifactStore:
    """Content-addressed filesystem store with verified reads and writes.

    Args:
        root: Directory that holds the blobs and the encryption state marker.
        policy: Optional storage policy.
        encryption: Required encryption configuration. Encrypted stores seal
            artifact bytes in a record-bound AEAD envelope before any byte,
            including the temporary file, reaches disk. Plaintext requires
            :meth:`JourneyStoreEncryption.development_plaintext`.

    Raises:
        JourneyEncryptionError: If the configuration is missing, refused by
            the active profile, or incompatible with the persisted state.
        LocalStoreError: If the directories cannot be prepared.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        policy: StoragePolicy | None = None,
        encryption: JourneyStoreEncryption | None = None,
    ) -> None:
        self._encryption = require_journey_encryption(encryption)
        self.root = Path(root)
        self.policy = policy or AllowAllStoragePolicy()
        self._lock = threading.RLock()
        self._codec: PlaintextJourneyCodec | EncryptedJourneyCodec = (
            PLAINTEXT_JOURNEY_CODEC
        )
        self._marker_state: PersistedEncryptionState | None = None
        self._key_verifiers: dict[str, str] = {}
        self._prepare_directories()
        self._codec = self._bind_encryption()
        if isinstance(self._codec, EncryptedJourneyCodec):
            try:
                self._codec.check_active_key()
            except OSError as exc:
                raise LocalStoreError(
                    "local artifact store cannot be initialized"
                ) from exc

    @property
    def encryption_state(self) -> str:
        """Return ``plaintext``, ``migrating``, or ``encrypted``."""

        return self._codec.state

    @classmethod
    def migrate_to_encrypted(
        cls,
        root: str | Path,
        *,
        encryption: JourneyStoreEncryption,
        policy: StoragePolicy | None = None,
        max_files: int | None = None,
    ) -> StoreResult[ReencryptionReport]:
        """Seal every plaintext blob in place; safe to resume after a crash.

        Each blob is replaced atomically, so an interrupted run leaves every
        file either plaintext or sealed. Re-running resumes from the files
        still in plaintext. Stale temporary files are removed.

        Args:
            root: Artifact store root.
            encryption: Encrypted target configuration.
            policy: Optional storage policy.
            max_files: Optional bound on files rewritten in this call.

        Returns:
            A value-free report, or a typed failure.
        """

        if not isinstance(encryption, JourneyStoreEncryption):
            raise TypeError("encryption must be a JourneyStoreEncryption")
        try:
            store = cls(
                root,
                policy=policy,
                encryption=replace(encryption, migrate_plaintext=True),
            )
        except EncryptionError as exc:
            return StoreResult.outcome(StoreState.FAILURE, exc.code)
        except (LocalStoreError, ValueError):
            return StoreResult.outcome(StoreState.FAILURE, "store_open_failed")
        return store.reencrypt(max_files=max_files)

    def put_bytes(
        self,
        artifact: ClinicalArtifact,
        content: bytes,
    ) -> StoreResult[ClinicalArtifact]:
        """Persist bytes only when size and content digest match metadata."""

        if not self.policy.allows("write", "artifact"):
            return StoreResult.outcome(StoreState.DENIED, "policy_denied")
        if not isinstance(content, bytes):
            return StoreResult.outcome(StoreState.FAILURE, "invalid_content_type")
        if len(content) != artifact.byte_size:
            return StoreResult.outcome(StoreState.CONFLICT, "artifact_size_mismatch")
        if sha256_digest(content) != artifact.content_hash:
            return StoreResult.outcome(StoreState.CONFLICT, "artifact_hash_mismatch")

        target = self._blob_path(artifact.content_hash)
        with self._lock:
            existing = self._read_verified(artifact.content_hash)
            if existing.ok:
                return StoreResult.success(artifact, created=False)
            if existing.state not in {StoreState.UNKNOWN}:
                return StoreResult.outcome(
                    existing.state, existing.code or "read_failed"
                )
            try:
                stored = self._codec.seal_blob(artifact.content_hash, content)
            except EncryptionError as exc:
                return StoreResult.outcome(StoreState.FAILURE, exc.code)
            except OSError:
                return StoreResult.outcome(StoreState.FAILURE, "artifact_write_failed")
            try:
                target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                os.chmod(target.parent, 0o700)
                descriptor, temp_name = tempfile.mkstemp(
                    prefix=_BLOB_TEMP_PREFIX,
                    dir=target.parent,
                )
                try:
                    if hasattr(os, "fchmod"):
                        os.fchmod(descriptor, 0o600)
                    with os.fdopen(descriptor, "wb") as stream:
                        stream.write(stored)
                        stream.flush()
                        os.fsync(stream.fileno())
                    try:
                        os.link(temp_name, target)
                        created = True
                    except FileExistsError:
                        created = False
                    finally:
                        Path(temp_name).unlink(missing_ok=True)
                    _fsync_directory(target.parent)
                except BaseException:
                    try:
                        os.close(descriptor)
                    except OSError:
                        pass
                    Path(temp_name).unlink(missing_ok=True)
                    raise
            except OSError:
                return StoreResult.outcome(StoreState.FAILURE, "artifact_write_failed")

            verified = self._read_verified(artifact.content_hash)
            if not verified.ok:
                if created:
                    target.unlink(missing_ok=True)
                return StoreResult.outcome(StoreState.FAILURE, "artifact_verify_failed")
            return StoreResult.success(artifact, created=created)

    def get_bytes(self, content_hash: str) -> StoreResult[bytes]:
        """Read bytes and revalidate the content-addressed digest."""

        if not self.policy.allows("read", "artifact"):
            return StoreResult.outcome(StoreState.DENIED, "policy_denied")
        return self._read_verified(content_hash)

    def reencrypt(
        self,
        *,
        max_files: int | None = None,
    ) -> StoreResult[ReencryptionReport]:
        """Re-seal blobs that are plaintext or sealed under an older key.

        Run after rotating the provider's active key, before retiring the
        previous key. Every rewrite is an atomic same-directory replace, so a
        crash leaves each blob readable with either the previous or the
        active key; re-running resumes. Content digests never change.

        Args:
            max_files: Optional bound on files rewritten in this call.

        Returns:
            A value-free report, or a typed failure whose ``value`` carries
            the progress made before the failure.
        """

        if not self.policy.allows("reencrypt", "artifact"):
            return StoreResult.outcome(StoreState.DENIED, "policy_denied")
        _require_positive_bound(max_files, "max_files")
        codec = self._codec
        if not isinstance(codec, EncryptedJourneyCodec):
            return StoreResult.outcome(StoreState.FAILURE, "store_not_encrypted")
        progress = _Progress()
        with self._lock:
            try:
                active_key_id = codec.active_key_id()
                if codec.state == ENCRYPTION_STATE_MIGRATING:
                    self._remove_stale_temporary_files()
                for path in self._blob_paths():
                    if max_files is not None and progress.rewritten >= max_files:
                        progress.complete = False
                        break
                    progress.examined += 1
                    content_hash = "sha256:" + path.name
                    stored = path.read_bytes()
                    current_key_id = codec.blob_key_id(content_hash, stored)
                    if current_key_id == active_key_id:
                        continue
                    content = codec.open_blob(content_hash, stored)
                    if sha256_digest(content) != content_hash:
                        return progress.failure(
                            "artifact_integrity_failed", codec.state
                        )
                    _replace_file(path, codec.seal_blob(content_hash, content))
                    progress.count(migrated=current_key_id is None)
                if progress.complete and codec.state == ENCRYPTION_STATE_MIGRATING:
                    self._write_marker(encrypted_state(codec))
                    self._codec = codec.completed()
            except EncryptionError as exc:
                return progress.failure(exc.code, codec.state)
            except OSError:
                return progress.failure("artifact_reencrypt_failed", codec.state)
            return StoreResult.success(progress.report(self._codec.state))

    def _read_verified(self, content_hash: str) -> StoreResult[bytes]:
        """Read and verify a blob for internal integrity operations."""

        try:
            target = self._blob_path(content_hash)
        except ValueError:
            return StoreResult.outcome(StoreState.FAILURE, "invalid_content_hash")
        try:
            if target.is_symlink():
                return StoreResult.outcome(StoreState.FAILURE, "artifact_path_unsafe")
            stored = target.read_bytes()
        except FileNotFoundError:
            return StoreResult.outcome(StoreState.UNKNOWN, "artifact_not_found")
        except OSError:
            return StoreResult.outcome(StoreState.FAILURE, "artifact_read_failed")
        try:
            content = self._codec.open_blob(content_hash, stored)
        except EncryptionError as exc:
            return StoreResult.outcome(StoreState.FAILURE, exc.code)
        if sha256_digest(content) != content_hash:
            return StoreResult.outcome(StoreState.FAILURE, "artifact_integrity_failed")
        return StoreResult.success(content)

    def discard_if_created(self, content_hash: str) -> None:
        """Remove a newly staged blob after a metadata transaction rolls back."""

        try:
            target = self._blob_path(content_hash)
            if target.is_file() and not target.is_symlink():
                target.unlink()
                _fsync_directory(target.parent)
        except (OSError, ValueError):
            return

    def _prepare_directories(self) -> None:
        try:
            self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
            os.chmod(self.root, 0o700)
            blob_root = self.root / "blobs" / "sha256"
            blob_root.mkdir(mode=0o700, parents=True, exist_ok=True)
            os.chmod(blob_root, 0o700)
        except OSError as exc:
            raise LocalStoreError("local artifact store cannot be initialized") from exc

    def _bind_encryption(self) -> PlaintextJourneyCodec | EncryptedJourneyCodec:
        try:
            current = self._read_marker()
            has_data = next(iter(self._blob_paths()), None) is not None
            binding = bind_encryption_state(
                self._encryption,
                current,
                has_data=has_data,
                verifiers=_MarkerKeyVerifiers(self),
            )
            if binding.persist is not None:
                self._write_marker(binding.persist)
        except OSError as exc:
            raise LocalStoreError("local artifact store cannot be initialized") from exc
        return binding.codec

    def _read_marker(self) -> PersistedEncryptionState | None:
        marker = self.root / _ENCRYPTION_MARKER
        if marker.is_symlink():
            raise JourneyEncryptionError("encryption_state_invalid")
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise JourneyEncryptionError("encryption_state_invalid") from None
        if (
            not isinstance(payload, dict)
            or set(payload) != _MARKER_FIELDS
            or payload["schema_version"] != _ENCRYPTION_MARKER_SCHEMA_VERSION
            or not isinstance(payload["key_verifiers"], dict)
            or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in payload["key_verifiers"].items()
            )
        ):
            raise JourneyEncryptionError("encryption_state_invalid")
        state = PersistedEncryptionState(
            state=payload["state"],
            key_purpose=payload["key_purpose"],
            envelope_version=payload["envelope_version"],
        )
        self._marker_state = state
        self._key_verifiers = dict(payload["key_verifiers"])
        return state

    def _write_marker(self, state: PersistedEncryptionState) -> None:
        document = {
            **state.to_dict(),
            "key_verifiers": dict(sorted(self._key_verifiers.items())),
            "schema_version": _ENCRYPTION_MARKER_SCHEMA_VERSION,
        }
        encoded = json.dumps(document, sort_keys=True, separators=(",", ":"))
        _replace_file(self.root / _ENCRYPTION_MARKER, encoded.encode("ascii"))
        self._marker_state = state

    def _save_key_verifier(self, key_id: str, verifier: str) -> None:
        with self._lock:
            if key_id in self._key_verifiers or self._marker_state is None:
                return
            self._key_verifiers[key_id] = verifier
            try:
                self._write_marker(self._marker_state)
            except BaseException:
                self._key_verifiers.pop(key_id, None)
                raise

    def _blob_paths(self) -> Iterator[Path]:
        blob_root = self.root / "blobs" / "sha256"
        for prefix in sorted(blob_root.iterdir()):
            if prefix.is_symlink() or not prefix.is_dir():
                continue
            for candidate in sorted(prefix.iterdir()):
                if (
                    _BLOB_NAME_RE.fullmatch(candidate.name) is not None
                    and candidate.name.startswith(prefix.name)
                    and candidate.is_file()
                    and not candidate.is_symlink()
                ):
                    yield candidate

    def _remove_stale_temporary_files(self) -> None:
        blob_root = self.root / "blobs" / "sha256"
        for prefix in sorted(blob_root.iterdir()):
            if prefix.is_symlink() or not prefix.is_dir():
                continue
            for candidate in sorted(prefix.iterdir()):
                if (
                    candidate.name.startswith(_BLOB_TEMP_PREFIX)
                    and candidate.is_file()
                    and not candidate.is_symlink()
                ):
                    candidate.unlink(missing_ok=True)
            _fsync_directory(prefix)

    def _blob_path(self, content_hash: str) -> Path:
        if (
            not isinstance(content_hash, str)
            or _DIGEST_RE.fullmatch(content_hash) is None
        ):
            raise ValueError("content hash must be a normalized SHA-256 digest")
        digest = content_hash.removeprefix("sha256:")
        return self.root / "blobs" / "sha256" / digest[:2] / digest


class SQLiteJourneyStore:
    """Append-only SQLite store with revisions and deterministic migrations.

    Args:
        path: SQLite database path.
        policy: Optional storage policy.
        encryption: Required encryption configuration. Encrypted stores seal
            every ``payload_json`` value in a record-bound AEAD envelope before
            SQLite sees it, so the database, its write-ahead log, and SQLite
            temporary storage only receive ciphertext for payloads. Plaintext
            requires :meth:`JourneyStoreEncryption.development_plaintext`.

    Raises:
        JourneyEncryptionError: If the configuration is missing, refused by
            the active profile, or incompatible with the persisted state.
        StoreCompatibilityError: If the persisted schema is unsupported.
        LocalStoreError: If the database cannot be opened.
    """

    _payload_codec: JourneyPayloadCodec = PLAINTEXT_JOURNEY_CODEC

    def __init__(
        self,
        path: str | Path,
        *,
        policy: StoragePolicy | None = None,
        encryption: JourneyStoreEncryption | None = None,
    ) -> None:
        assert_contract_compatibility()
        self._encryption = require_journey_encryption(encryption)
        self.path = Path(path)
        self.policy = policy or AllowAllStoragePolicy()
        self._lock = threading.RLock()
        self._closed = False
        self._prepare_path()
        try:
            self._connection = sqlite3.connect(
                self.path,
                isolation_level=None,
                check_same_thread=False,
            )
            os.chmod(self.path, 0o600)
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute("PRAGMA journal_mode = WAL")
            self._connection.execute("PRAGMA synchronous = FULL")
            self._connection.execute("PRAGMA busy_timeout = 5000")
            # Zero freed cells and keep transient tables and sorts in memory.
            self._connection.execute("PRAGMA secure_delete = ON")
            self._connection.execute("PRAGMA temp_store = MEMORY")
            self._apply_migrations()
            self._payload_codec = self._bind_encryption()
            if isinstance(self._payload_codec, EncryptedJourneyCodec):
                self._payload_codec.check_active_key()
        except (StoreCompatibilityError, EncryptionError):
            self._close_after_failed_open()
            raise
        except (OSError, sqlite3.Error) as exc:
            self._close_after_failed_open()
            raise LocalStoreError("local metadata store cannot be initialized") from exc

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        policy: StoragePolicy | None = None,
        encryption: JourneyStoreEncryption | None = None,
    ) -> StoreResult["SQLiteJourneyStore"]:
        """Open a store with typed compatibility and failure outcomes."""

        try:
            return StoreResult.success(cls(path, policy=policy, encryption=encryption))
        except StoreCompatibilityError:
            return StoreResult.outcome(StoreState.UNSUPPORTED, "schema_unsupported")
        except EncryptionError as exc:
            return StoreResult.outcome(StoreState.FAILURE, exc.code)
        except LocalStoreError:
            return StoreResult.outcome(StoreState.FAILURE, "store_open_failed")

    @classmethod
    def migrate_to_encrypted(
        cls,
        path: str | Path,
        *,
        encryption: JourneyStoreEncryption,
        policy: StoragePolicy | None = None,
        batch_size: int = 256,
        max_batches: int | None = None,
    ) -> StoreResult[ReencryptionReport]:
        """Encrypt every payload of a plaintext store in resumable batches.

        The store is first marked ``migrating``; from then on it only opens
        with an encrypted configuration. Each batch commits atomically, so an
        interrupted migration rolls back the batch in flight and resumes from
        the remaining plaintext rows when called again. On completion the
        store is marked ``encrypted``, vacuumed, and its write-ahead log is
        truncated so earlier plaintext pages are not left behind in the
        database files.

        Args:
            path: SQLite database path.
            encryption: Encrypted target configuration.
            policy: Optional storage policy.
            batch_size: Rows per committed batch.
            max_batches: Optional bound on committed batches in this call.

        Returns:
            A value-free report, or a typed failure.
        """

        if not isinstance(encryption, JourneyStoreEncryption):
            raise TypeError("encryption must be a JourneyStoreEncryption")
        try:
            store = cls(
                path,
                policy=policy,
                encryption=replace(encryption, migrate_plaintext=True),
            )
        except StoreCompatibilityError:
            return StoreResult.outcome(StoreState.UNSUPPORTED, "schema_unsupported")
        except EncryptionError as exc:
            return StoreResult.outcome(StoreState.FAILURE, exc.code)
        except (LocalStoreError, ValueError):
            return StoreResult.outcome(StoreState.FAILURE, "store_open_failed")
        try:
            return store.reencrypt(batch_size=batch_size, max_batches=max_batches)
        finally:
            store.close()

    @property
    def encryption_state(self) -> str:
        """Return ``plaintext``, ``migrating``, or ``encrypted``."""

        return self._payload_codec.state

    def reencrypt(
        self,
        *,
        batch_size: int = 256,
        max_batches: int | None = None,
    ) -> StoreResult[ReencryptionReport]:
        """Re-seal payloads that are plaintext or sealed under an older key.

        Run after rotating the provider's active key and before retiring the
        previous key. Rows keep their identifiers, payload hashes, and
        revisions, so history and point-in-time reads are unchanged. Each
        batch commits atomically; re-running resumes. While a plaintext store
        is migrating, completion also marks it ``encrypted`` and compacts it.

        Args:
            batch_size: Rows per committed batch.
            max_batches: Optional bound on committed batches in this call.

        Returns:
            A value-free report, or a typed failure whose ``value`` carries
            the progress made before the failure.
        """

        denied = self._denied("reencrypt", "store")
        if denied is not None:
            return denied
        _require_positive_bound(batch_size, "batch_size")
        _require_positive_bound(max_batches, "max_batches")
        self._require_open()
        codec = self._payload_codec
        if not isinstance(codec, EncryptedJourneyCodec):
            return StoreResult.outcome(StoreState.FAILURE, "store_not_encrypted")
        progress = _Progress()
        with self._lock:
            try:
                active_key_id = codec.active_key_id()
                batches = 0
                for table in PAYLOAD_RECORD_KEYS:
                    after_row: int | None = 0
                    while after_row is not None:
                        if max_batches is not None and batches >= max_batches:
                            progress.complete = False
                            break
                        wrote, after_row = self._reencrypt_batch(
                            codec,
                            table,
                            active_key_id=active_key_id,
                            after_row=after_row,
                            batch_size=batch_size,
                            progress=progress,
                        )
                        batches += int(wrote)
                    if not progress.complete:
                        break
                if progress.complete and codec.state == ENCRYPTION_STATE_MIGRATING:
                    self._complete_migration(codec)
                elif progress.complete:
                    self._connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except EncryptionError as exc:
                return progress.failure(exc.code, codec.state)
            except _PayloadHashMismatch:
                return progress.failure("payload_hash_mismatch", codec.state)
            except (sqlite3.Error, ValueError, TypeError):
                return progress.failure("reencrypt_failed", codec.state)
            return StoreResult.success(progress.report(self._payload_codec.state))

    def _reencrypt_batch(
        self,
        codec: EncryptedJourneyCodec,
        table: str,
        *,
        active_key_id: str,
        after_row: int,
        batch_size: int,
        progress: "_Progress",
    ) -> tuple[bool, int | None]:
        key_columns = ", ".join(PAYLOAD_RECORD_KEYS[table])
        migrated = reencrypted = 0
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            rows = self._connection.execute(
                f"SELECT rowid AS row_ref, {key_columns}, payload_hash, payload_json "  # noqa: S608
                f"FROM {table} WHERE rowid > ? ORDER BY rowid LIMIT ?",
                (after_row, batch_size),
            ).fetchall()
            updates: list[tuple[str, int]] = []
            for row in rows:
                current_key_id = codec.payload_key_id(row["payload_json"])
                if current_key_id == active_key_id:
                    continue
                identity = _record_identity(table, row)
                payload_json = codec.open_payload(table, identity, row["payload_json"])
                if canonical_digest(json.loads(payload_json)) != row["payload_hash"]:
                    raise _PayloadHashMismatch
                updates.append(
                    (
                        codec.seal_payload(table, identity, payload_json),
                        int(row["row_ref"]),
                    )
                )
                if current_key_id is None:
                    migrated += 1
                else:
                    reencrypted += 1
            if updates:
                self._connection.executemany(
                    f"UPDATE {table} SET payload_json = ? WHERE rowid = ?",  # noqa: S608
                    updates,
                )
            self._connection.execute("COMMIT")
        except BaseException:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise
        progress.examined += len(rows)
        progress.migrated += migrated
        progress.reencrypted += reencrypted
        next_row = int(rows[-1]["row_ref"]) if len(rows) == batch_size else None
        return bool(updates), next_row

    def _complete_migration(self, codec: EncryptedJourneyCodec) -> None:
        # Rebuild the database so pages written before encryption are not
        # retained, then truncate the write-ahead log that held them.
        self._connection.execute("VACUUM")
        self._connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            self._write_encryption_state(encrypted_state(codec))
            self._connection.execute("COMMIT")
        except BaseException:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise
        self._connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self._payload_codec = codec.completed()

    def _bind_encryption(self) -> JourneyPayloadCodec:
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            row = self._connection.execute(
                "SELECT state, key_purpose, envelope_version "
                "FROM store_encryption WHERE singleton = 1"
            ).fetchone()
            current = (
                None
                if row is None
                else PersistedEncryptionState(
                    state=row["state"],
                    key_purpose=row["key_purpose"],
                    envelope_version=row["envelope_version"],
                )
            )
            has_data = any(
                self._connection.execute(
                    f"SELECT 1 FROM {table} LIMIT 1"  # noqa: S608
                ).fetchone()
                is not None
                for table in PAYLOAD_RECORD_KEYS
            )
            binding = bind_encryption_state(
                self._encryption,
                current,
                has_data=has_data,
                verifiers=_SQLiteKeyVerifiers(self._connection),
            )
            if binding.persist is not None:
                self._write_encryption_state(binding.persist)
            self._connection.execute("COMMIT")
        except BaseException:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise
        return binding.codec

    def _write_encryption_state(self, state: PersistedEncryptionState) -> None:
        self._connection.execute(
            """
            INSERT INTO store_encryption(
                singleton, state, key_purpose, envelope_version
            ) VALUES (1, ?, ?, ?)
            ON CONFLICT(singleton) DO UPDATE SET
                state = excluded.state,
                key_purpose = excluded.key_purpose,
                envelope_version = excluded.envelope_version
            """,
            (state.state, state.key_purpose, state.envelope_version),
        )

    def _decode_payload(self, table: str, row: Any) -> str:
        """Return canonical payload JSON for one row selected with its keys."""

        return self._payload_codec.open_payload(
            table,
            _record_identity(table, row),
            row["payload_json"],
        )

    def _encode_payload(
        self,
        table: str,
        identity: tuple[str | int, ...],
        payload_json: str,
    ) -> str:
        """Return the persisted form of one payload for its record identity."""

        return self._payload_codec.seal_payload(table, identity, payload_json)

    @property
    def schema_version(self) -> int:
        """Return the latest verified migration version."""

        row = self._connection.execute(
            "SELECT COALESCE(MAX(version), 0) AS version FROM schema_migrations"
        ).fetchone()
        return int(row["version"])

    @property
    def latest_revision(self) -> int | None:
        """Return the latest committed revision, if any."""

        row = self._connection.execute(
            "SELECT MAX(revision) AS revision FROM store_revisions"
        ).fetchone()
        return None if row["revision"] is None else int(row["revision"])

    def close(self) -> None:
        """Checkpoint and close the local database."""

        with self._lock:
            if self._closed:
                return
            try:
                self._connection.execute("PRAGMA wal_checkpoint(FULL)")
            finally:
                self._connection.close()
                self._closed = True

    def __enter__(self) -> "SQLiteJourneyStore":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    @contextmanager
    def transaction(self, *, committed_at: str) -> Iterator["SQLiteJourneyTransaction"]:
        """Open one atomic append-only transaction at a supplied timestamp."""

        if not _valid_timestamp(committed_at):
            raise ValueError("committed_at must be a timezone-aware ISO timestamp")
        self._require_open()
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                cursor = self._connection.execute(
                    "INSERT INTO store_revisions(committed_at) VALUES (?)",
                    (committed_at,),
                )
                if cursor.lastrowid is None:
                    raise LocalStoreError("local store revision was not allocated")
                transaction = SQLiteJourneyTransaction(self, int(cursor.lastrowid))
                yield transaction
                if transaction.failed or not transaction.mutated:
                    self._connection.execute("ROLLBACK")
                else:
                    self._connection.execute("COMMIT")
            except BaseException:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise

    def put_artifact(
        self,
        artifact: ClinicalArtifact,
        *,
        committed_at: str,
    ) -> StoreResult[ClinicalArtifact]:
        """Persist artifact metadata in one auto-committed revision."""

        return self._auto_put(committed_at, lambda tx: tx.put_artifact(artifact))

    def put_evidence(
        self,
        locator: EvidenceLocator,
        *,
        committed_at: str,
    ) -> StoreResult[EvidenceLocator]:
        """Persist evidence in one auto-committed revision."""

        return self._auto_put(committed_at, lambda tx: tx.put_evidence(locator))

    def put_fact(
        self,
        fact: ClinicalFact,
        *,
        committed_at: str,
    ) -> StoreResult[ClinicalFact]:
        """Persist a fact in one auto-committed revision."""

        return self._auto_put(committed_at, lambda tx: tx.put_fact(fact))

    def put_conflict(
        self,
        conflict: ConflictSet,
        *,
        committed_at: str,
    ) -> StoreResult[ConflictSet]:
        """Persist a conflict set in one auto-committed revision."""

        return self._auto_put(committed_at, lambda tx: tx.put_conflict(conflict))

    def put_resolution(
        self,
        resolution: ResolutionEvent,
        *,
        committed_at: str,
    ) -> StoreResult[ResolutionEvent]:
        """Persist a resolution event in one auto-committed revision."""

        return self._auto_put(committed_at, lambda tx: tx.put_resolution(resolution))

    def put_dataset(
        self,
        snapshot: DatasetSnapshot,
        *,
        committed_at: str,
    ) -> StoreResult[DatasetSnapshot]:
        """Persist a dataset snapshot in one auto-committed revision."""

        return self._auto_put(committed_at, lambda tx: tx.put_dataset(snapshot))

    def put_canonical(
        self,
        record: CanonicalRecord,
        *,
        committed_at: str,
    ) -> StoreResult[CanonicalRecordVersion]:
        """Append a canonical-record version in one revision."""

        return self._auto_put(committed_at, lambda tx: tx.put_canonical(record))

    def put_job(
        self,
        job: JobMetadata,
        *,
        committed_at: str,
    ) -> StoreResult[JobMetadata]:
        """Append PHI-free job metadata in one revision."""

        return self._auto_put(committed_at, lambda tx: tx.put_job(job))

    def get_artifact(
        self,
        artifact_id: str,
        *,
        as_of: StorePoint | None = None,
    ) -> StoreResult[ClinicalArtifact]:
        """Read artifact metadata visible at one revision."""

        return self._read_record(
            operation="read",
            record_type="artifact",
            table="artifacts",
            id_column="artifact_id",
            record_id=artifact_id,
            parser=ClinicalArtifact.from_json,
            as_of=as_of,
        )

    def get_artifact_by_hash(
        self,
        content_hash: str,
        *,
        as_of: StorePoint | None = None,
    ) -> StoreResult[ClinicalArtifact]:
        """Read artifact metadata by a normalized content digest."""

        if _DIGEST_RE.fullmatch(content_hash) is None:
            return StoreResult.outcome(StoreState.FAILURE, "invalid_content_hash")
        return self._read_record(
            operation="read",
            record_type="artifact",
            table="artifacts",
            id_column="content_hash",
            record_id=content_hash,
            parser=ClinicalArtifact.from_json,
            as_of=as_of,
        )

    def get_evidence(
        self,
        locator_id: str,
        *,
        as_of: StorePoint | None = None,
    ) -> StoreResult[EvidenceLocator]:
        """Read evidence visible at one revision."""

        return self._read_record(
            operation="read",
            record_type="evidence",
            table="evidence_locators",
            id_column="locator_id",
            record_id=locator_id,
            parser=EvidenceLocator.from_json,
            as_of=as_of,
        )

    def get_fact(
        self,
        fact_id: str,
        *,
        as_of: StorePoint | None = None,
    ) -> StoreResult[ClinicalFact]:
        """Read a fact visible at one revision."""

        return self._read_record(
            operation="read",
            record_type="fact",
            table="clinical_facts",
            id_column="fact_id",
            record_id=fact_id,
            parser=ClinicalFact.from_json,
            as_of=as_of,
        )

    def get_conflict(
        self,
        conflict_id: str,
        *,
        as_of: StorePoint | None = None,
    ) -> StoreResult[ConflictSet]:
        """Read a conflict visible at one revision."""

        return self._read_record(
            operation="read",
            record_type="conflict",
            table="conflict_sets",
            id_column="conflict_id",
            record_id=conflict_id,
            parser=ConflictSet.from_json,
            as_of=as_of,
        )

    def get_resolution(
        self,
        resolution_id: str,
        *,
        as_of: StorePoint | None = None,
    ) -> StoreResult[ResolutionEvent]:
        """Read a resolution visible at one revision."""

        return self._read_record(
            operation="read",
            record_type="resolution",
            table="resolution_events",
            id_column="resolution_id",
            record_id=resolution_id,
            parser=ResolutionEvent.from_json,
            as_of=as_of,
        )

    def get_dataset(
        self,
        snapshot_id: str,
        *,
        as_of: StorePoint | None = None,
    ) -> StoreResult[DatasetSnapshot]:
        """Read a dataset snapshot visible at one revision."""

        return self._read_record(
            operation="read",
            record_type="dataset",
            table="dataset_snapshots",
            id_column="snapshot_id",
            record_id=snapshot_id,
            parser=DatasetSnapshot.from_json,
            as_of=as_of,
        )

    def get_canonical(
        self,
        canonical_id: str,
        *,
        as_of: StorePoint | None = None,
    ) -> StoreResult[CanonicalRecordVersion]:
        """Read the newest canonical-record version visible at one revision."""

        denied = self._denied("read", "canonical")
        if denied is not None:
            return denied
        if not _valid_opaque_id(canonical_id):
            return StoreResult.outcome(StoreState.FAILURE, "invalid_identifier")
        cutoff = self._cutoff(as_of)
        row = self._connection.execute(
            """
            SELECT canonical_id, payload_json, version, created_revision
            FROM canonical_record_versions
            WHERE canonical_id = ? AND created_revision <= ?
            ORDER BY created_revision DESC, version DESC
            LIMIT 1
            """,
            (canonical_id, cutoff),
        ).fetchone()
        if row is None:
            return StoreResult.outcome(StoreState.UNKNOWN, "canonical_not_found")
        try:
            record = CanonicalRecord.from_dict(
                json.loads(self._decode_payload("canonical_record_versions", row))
            )
            version = CanonicalRecordVersion(
                record=record,
                version=int(row["version"]),
                revision=int(row["created_revision"]),
            )
        except EncryptionError as exc:
            return StoreResult.outcome(StoreState.FAILURE, exc.code)
        except (ValueError, TypeError, json.JSONDecodeError):
            return StoreResult.outcome(StoreState.FAILURE, "stored_payload_invalid")
        return StoreResult.success(version, revision=version.revision)

    def list_canonical_versions(
        self,
        canonical_id: str,
        *,
        as_of: StorePoint | None = None,
    ) -> StoreResult[tuple[CanonicalRecordVersion, ...]]:
        """Return complete append-only canonical history through one revision."""

        denied = self._denied("read", "canonical")
        if denied is not None:
            return denied
        if not _valid_opaque_id(canonical_id):
            return StoreResult.outcome(StoreState.FAILURE, "invalid_identifier")
        rows = self._connection.execute(
            """
            SELECT canonical_id, payload_json, version, created_revision
            FROM canonical_record_versions
            WHERE canonical_id = ? AND created_revision <= ?
            ORDER BY version
            """,
            (canonical_id, self._cutoff(as_of)),
        ).fetchall()
        if not rows:
            return StoreResult.outcome(StoreState.UNKNOWN, "canonical_not_found")
        try:
            versions = tuple(
                CanonicalRecordVersion(
                    record=CanonicalRecord.from_dict(
                        json.loads(
                            self._decode_payload("canonical_record_versions", row)
                        )
                    ),
                    version=int(row["version"]),
                    revision=int(row["created_revision"]),
                )
                for row in rows
            )
        except EncryptionError as exc:
            return StoreResult.outcome(StoreState.FAILURE, exc.code)
        except (ValueError, TypeError, json.JSONDecodeError):
            return StoreResult.outcome(StoreState.FAILURE, "stored_payload_invalid")
        return StoreResult.success(versions)

    def list_canonical_records(
        self,
        subject_id: str,
        *,
        as_of: StorePoint | None = None,
    ) -> StoreResult[tuple[CanonicalRecordVersion, ...]]:
        """Return each latest visible canonical pointer for one subject."""

        denied = self._denied("read", "canonical")
        if denied is not None:
            return denied
        if not _valid_opaque_id(subject_id):
            return StoreResult.outcome(StoreState.FAILURE, "invalid_identifier")
        cutoff = self._cutoff(as_of)
        rows = self._connection.execute(
            """
            SELECT visible_record.canonical_id, visible_record.payload_json,
                   visible_record.version, visible_record.created_revision
            FROM canonical_record_versions AS visible_record
            WHERE visible_record.subject_id = ?
              AND visible_record.created_revision <= ?
              AND visible_record.version = (
                  SELECT MAX(candidate.version)
                  FROM canonical_record_versions AS candidate
                  WHERE candidate.canonical_id = visible_record.canonical_id
                    AND candidate.created_revision <= ?
              )
            ORDER BY visible_record.canonical_id
            """,
            (subject_id, cutoff, cutoff),
        ).fetchall()
        try:
            records = tuple(
                CanonicalRecordVersion(
                    record=CanonicalRecord.from_dict(
                        json.loads(
                            self._decode_payload("canonical_record_versions", row)
                        )
                    ),
                    version=int(row["version"]),
                    revision=int(row["created_revision"]),
                )
                for row in rows
            )
        except EncryptionError as exc:
            return StoreResult.outcome(StoreState.FAILURE, exc.code)
        except (ValueError, TypeError, json.JSONDecodeError):
            return StoreResult.outcome(StoreState.FAILURE, "stored_payload_invalid")
        return StoreResult.success(records)

    def get_job(
        self,
        job_id: str,
        *,
        as_of: StorePoint | None = None,
    ) -> StoreResult[JobMetadata]:
        """Read the newest job metadata visible at one revision."""

        denied = self._denied("read", "job")
        if denied is not None:
            return denied
        if not _valid_opaque_id(job_id):
            return StoreResult.outcome(StoreState.FAILURE, "invalid_identifier")
        row = self._connection.execute(
            """
            SELECT job_id, version, payload_json, created_revision
            FROM job_metadata_versions
            WHERE job_id = ? AND created_revision <= ?
            ORDER BY created_revision DESC, version DESC
            LIMIT 1
            """,
            (job_id, self._cutoff(as_of)),
        ).fetchone()
        if row is None:
            return StoreResult.outcome(StoreState.UNKNOWN, "job_not_found")
        try:
            job = JobMetadata.from_dict(
                json.loads(self._decode_payload("job_metadata_versions", row))
            )
        except EncryptionError as exc:
            return StoreResult.outcome(StoreState.FAILURE, exc.code)
        except (ValueError, TypeError, json.JSONDecodeError):
            return StoreResult.outcome(StoreState.FAILURE, "stored_payload_invalid")
        return StoreResult.success(job, revision=int(row["created_revision"]))

    def list_job_versions(
        self,
        job_id: str,
        *,
        as_of: StorePoint | None = None,
    ) -> StoreResult[tuple[JobMetadata, ...]]:
        """Return append-only job metadata versions through one revision."""

        denied = self._denied("read", "job")
        if denied is not None:
            return denied
        if not _valid_opaque_id(job_id):
            return StoreResult.outcome(StoreState.FAILURE, "invalid_identifier")
        rows = self._connection.execute(
            """
            SELECT job_id, version, payload_json
            FROM job_metadata_versions
            WHERE job_id = ? AND created_revision <= ?
            ORDER BY version, created_revision
            """,
            (job_id, self._cutoff(as_of)),
        ).fetchall()
        if not rows:
            return StoreResult.outcome(StoreState.UNKNOWN, "job_not_found")
        try:
            versions = tuple(
                JobMetadata.from_dict(
                    json.loads(self._decode_payload("job_metadata_versions", row))
                )
                for row in rows
            )
        except EncryptionError as exc:
            return StoreResult.outcome(StoreState.FAILURE, exc.code)
        except (ValueError, TypeError, json.JSONDecodeError):
            return StoreResult.outcome(StoreState.FAILURE, "stored_payload_invalid")
        return StoreResult.success(versions)

    def list_jobs(
        self,
        *,
        as_of: StorePoint | None = None,
    ) -> StoreResult[tuple[JobMetadata, ...]]:
        """Return the latest visible version of each PHI-free job record."""

        denied = self._denied("read", "job")
        if denied is not None:
            return denied
        cutoff = self._cutoff(as_of)
        rows = self._connection.execute(
            """
            SELECT visible_job.job_id, visible_job.version,
                   visible_job.payload_json
            FROM job_metadata_versions AS visible_job
            WHERE visible_job.created_revision <= ?
              AND visible_job.version = (
                  SELECT MAX(candidate.version)
                  FROM job_metadata_versions AS candidate
                  WHERE candidate.job_id = visible_job.job_id
                    AND candidate.created_revision <= ?
              )
            ORDER BY visible_job.job_id
            """,
            (cutoff, cutoff),
        ).fetchall()
        try:
            jobs = tuple(
                JobMetadata.from_dict(
                    json.loads(self._decode_payload("job_metadata_versions", row))
                )
                for row in rows
            )
        except EncryptionError as exc:
            return StoreResult.outcome(StoreState.FAILURE, exc.code)
        except (ValueError, TypeError, json.JSONDecodeError):
            return StoreResult.outcome(StoreState.FAILURE, "stored_payload_invalid")
        return StoreResult.success(jobs)

    def list_facts(
        self,
        subject_id: str,
        *,
        as_of: StorePoint | None = None,
    ) -> StoreResult[tuple[ClinicalFact, ...]]:
        """Return deterministic fact history for one opaque subject."""

        denied = self._denied("read", "fact")
        if denied is not None:
            return denied
        if not _valid_opaque_id(subject_id):
            return StoreResult.outcome(StoreState.FAILURE, "invalid_identifier")
        rows = self._connection.execute(
            """
            SELECT fact_id, payload_json
            FROM clinical_facts
            WHERE subject_id = ? AND created_revision <= ?
            ORDER BY created_revision, fact_id
            """,
            (subject_id, self._cutoff(as_of)),
        ).fetchall()
        try:
            facts = tuple(
                ClinicalFact.from_json(self._decode_payload("clinical_facts", row))
                for row in rows
            )
        except EncryptionError as exc:
            return StoreResult.outcome(StoreState.FAILURE, exc.code)
        except ValueError:
            return StoreResult.outcome(StoreState.FAILURE, "stored_payload_invalid")
        return StoreResult.success(facts)

    def list_conflicts(
        self,
        subject_id: str,
        *,
        as_of: StorePoint | None = None,
    ) -> StoreResult[tuple[ConflictSet, ...]]:
        """Return deterministic conflict history for one opaque subject."""

        denied = self._denied("read", "conflict")
        if denied is not None:
            return denied
        if not _valid_opaque_id(subject_id):
            return StoreResult.outcome(StoreState.FAILURE, "invalid_identifier")
        rows = self._connection.execute(
            """
            SELECT conflict_id, payload_json
            FROM conflict_sets
            WHERE subject_id = ? AND created_revision <= ?
            ORDER BY created_revision, conflict_id
            """,
            (subject_id, self._cutoff(as_of)),
        ).fetchall()
        try:
            conflicts = tuple(
                ConflictSet.from_json(self._decode_payload("conflict_sets", row))
                for row in rows
            )
        except EncryptionError as exc:
            return StoreResult.outcome(StoreState.FAILURE, exc.code)
        except ValueError:
            return StoreResult.outcome(StoreState.FAILURE, "stored_payload_invalid")
        return StoreResult.success(conflicts)

    def list_resolutions(
        self,
        conflict_id: str,
        *,
        as_of: StorePoint | None = None,
    ) -> StoreResult[tuple[ResolutionEvent, ...]]:
        """Return append-only resolution history for one conflict."""

        denied = self._denied("read", "resolution")
        if denied is not None:
            return denied
        if not _valid_opaque_id(conflict_id):
            return StoreResult.outcome(StoreState.FAILURE, "invalid_identifier")
        rows = self._connection.execute(
            """
            SELECT resolution_id, payload_json
            FROM resolution_events
            WHERE conflict_id = ? AND created_revision <= ?
            ORDER BY created_revision, resolution_id
            """,
            (conflict_id, self._cutoff(as_of)),
        ).fetchall()
        try:
            resolutions = tuple(
                ResolutionEvent.from_json(
                    self._decode_payload("resolution_events", row)
                )
                for row in rows
            )
        except EncryptionError as exc:
            return StoreResult.outcome(StoreState.FAILURE, exc.code)
        except ValueError:
            return StoreResult.outcome(StoreState.FAILURE, "stored_payload_invalid")
        return StoreResult.success(resolutions)

    def integrity_check(self) -> StoreResult[Mapping[str, int]]:
        """Run local SQLite and payload-hash integrity checks without values."""

        denied = self._denied("verify", "store")
        if denied is not None:
            return denied
        row = self._connection.execute("PRAGMA integrity_check").fetchone()
        if row is None or row[0] != "ok":
            return StoreResult.outcome(StoreState.FAILURE, "sqlite_integrity_failed")
        if self._connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
            return StoreResult.outcome(
                StoreState.FAILURE, "foreign_key_integrity_failed"
            )
        counts: dict[str, int] = {}
        for table, key_columns in PAYLOAD_RECORD_KEYS.items():
            rows = self._connection.execute(
                f"SELECT {', '.join(key_columns)}, payload_hash, payload_json "  # noqa: S608
                f"FROM {table}"
            ).fetchall()
            try:
                mismatched = any(
                    canonical_digest(json.loads(self._decode_payload(table, item)))
                    != item["payload_hash"]
                    for item in rows
                )
            except EncryptionError as exc:
                return StoreResult.outcome(StoreState.FAILURE, exc.code)
            except (ValueError, TypeError, json.JSONDecodeError):
                return StoreResult.outcome(StoreState.FAILURE, "stored_payload_invalid")
            if mismatched:
                return StoreResult.outcome(StoreState.FAILURE, "payload_hash_mismatch")
            counts[table] = len(rows)
        return StoreResult.success(counts)

    def _auto_put(
        self,
        committed_at: str,
        operation: Callable[["SQLiteJourneyTransaction"], StoreResult[T]],
    ) -> StoreResult[T]:
        result: StoreResult[T]
        try:
            with self.transaction(committed_at=committed_at) as transaction:
                result = operation(transaction)
        except CommitStatusUnknown:
            return StoreResult.outcome(StoreState.UNKNOWN, "commit_status_unknown")
        except EncryptionError as exc:
            return StoreResult.outcome(StoreState.FAILURE, exc.code)
        except (LocalStoreError, sqlite3.Error, ValueError, TypeError):
            return StoreResult.outcome(StoreState.FAILURE, "transaction_failed")
        if not result.ok:
            return result
        return StoreResult.success(
            result.value,  # type: ignore[arg-type]
            created=result.created,
            revision=transaction.revision if result.created else result.revision,
        )

    def _read_record(
        self,
        *,
        operation: str,
        record_type: str,
        table: str,
        id_column: str,
        record_id: str,
        parser: Callable[[str], T],
        as_of: StorePoint | None,
    ) -> StoreResult[T]:
        denied = self._denied(operation, record_type)
        if denied is not None:
            return denied
        if id_column != "content_hash" and not _valid_opaque_id(record_id):
            return StoreResult.outcome(StoreState.FAILURE, "invalid_identifier")
        row = self._connection.execute(
            f"""
            SELECT {", ".join(PAYLOAD_RECORD_KEYS[table])}, payload_json,
                   created_revision
            FROM {table}
            WHERE {id_column} = ? AND created_revision <= ?
            """,  # noqa: S608
            (record_id, self._cutoff(as_of)),
        ).fetchone()
        if row is None:
            return StoreResult.outcome(StoreState.UNKNOWN, f"{record_type}_not_found")
        try:
            record = parser(self._decode_payload(table, row))
        except EncryptionError as exc:
            return StoreResult.outcome(StoreState.FAILURE, exc.code)
        except (ValueError, TypeError, json.JSONDecodeError):
            return StoreResult.outcome(StoreState.FAILURE, "stored_payload_invalid")
        return StoreResult.success(record, revision=int(row["created_revision"]))

    def _cutoff(self, as_of: StorePoint | None) -> int:
        if as_of is not None:
            return as_of.revision
        return self.latest_revision or 0

    def _denied(self, operation: str, record_type: str) -> StoreResult[Any] | None:
        if self.policy.allows(operation, record_type):
            return None
        return StoreResult.outcome(StoreState.DENIED, "policy_denied")

    def _prepare_path(self) -> None:
        try:
            if not self.path.parent.exists():
                self.path.parent.mkdir(mode=0o700, parents=True)
                os.chmod(self.path.parent, 0o700)
        except OSError as exc:
            raise LocalStoreError("local metadata path cannot be initialized") from exc

    def _apply_migrations(self) -> None:
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                checksum TEXT NOT NULL
            )
            """
        )
        applied = {
            int(row["version"]): (str(row["name"]), str(row["checksum"]))
            for row in self._connection.execute(
                "SELECT version, name, checksum FROM schema_migrations ORDER BY version"
            )
        }
        if applied and max(applied) > LATEST_MIGRATION_VERSION:
            raise StoreCompatibilityError("persisted schema major is unsupported")
        known = {migration.version: migration for migration in MIGRATIONS}
        for version, (name, checksum) in applied.items():
            migration = known.get(version)
            if (
                migration is None
                or migration.name != name
                or migration.checksum != checksum
            ):
                raise StoreCompatibilityError("persisted migration checksum differs")

        try:
            self._connection.execute("BEGIN IMMEDIATE")
            for migration in MIGRATIONS:
                if migration.version in applied:
                    continue
                for statement in migration.statements:
                    self._connection.execute(statement)
                self._connection.execute(
                    "INSERT INTO schema_migrations(version, name, checksum) "
                    "VALUES (?, ?, ?)",
                    (migration.version, migration.name, migration.checksum),
                )
            self._connection.execute("COMMIT")
        except sqlite3.Error as exc:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise StoreMigrationError("local store migration failed") from exc

    def _close_after_failed_open(self) -> None:
        connection = getattr(self, "_connection", None)
        if connection is not None:
            connection.close()

    def _require_open(self) -> None:
        if self._closed:
            raise LocalStoreError("local metadata store is closed")


class SQLiteJourneyTransaction:
    """Mutation surface bound to one SQLite revision and transaction."""

    def __init__(self, store: SQLiteJourneyStore, revision: int) -> None:
        self.store = store
        self.revision = revision
        self.failed = False
        self.mutated = False

    def put_artifact(self, artifact: ClinicalArtifact) -> StoreResult[ClinicalArtifact]:
        """Persist artifact metadata idempotently."""

        denied = self._denied("write", "artifact")
        if denied is not None:
            return denied
        if not all(
            self._exists("artifacts", "artifact_id", item)
            for item in artifact.parent_artifact_ids
        ):
            return self._fail(StoreState.PARTIAL, "missing_parent_artifact")
        existing_hash = self.store._connection.execute(
            "SELECT artifact_id, payload_json, created_revision "
            "FROM artifacts WHERE content_hash = ?",
            (artifact.content_hash,),
        ).fetchone()
        if existing_hash is not None:
            return self._existing(
                existing_hash,
                artifact,
                ClinicalArtifact.from_json,
                table="artifacts",
            )
        return self._insert_simple(
            table="artifacts",
            id_column="artifact_id",
            record_id=artifact.artifact_id,
            payload_json=artifact.to_json(),
            extra_columns=("content_hash",),
            extra_values=(artifact.content_hash,),
            value=artifact,
        )

    def put_evidence(self, locator: EvidenceLocator) -> StoreResult[EvidenceLocator]:
        """Persist evidence after its artifact dependency exists."""

        denied = self._denied("write", "evidence")
        if denied is not None:
            return denied
        if not self._exists("artifacts", "artifact_id", locator.artifact_id):
            return self._fail(StoreState.PARTIAL, "missing_artifact")
        return self._insert_simple(
            table="evidence_locators",
            id_column="locator_id",
            record_id=locator.locator_id,
            payload_json=locator.to_json(),
            extra_columns=("artifact_id",),
            extra_values=(locator.artifact_id,),
            value=locator,
        )

    def put_fact(self, fact: ClinicalFact) -> StoreResult[ClinicalFact]:
        """Persist a fact after evidence and parent dependencies exist."""

        denied = self._denied("write", "fact")
        if denied is not None:
            return denied
        if not all(
            self._exists("evidence_locators", "locator_id", item)
            for item in fact.evidence_ids
        ):
            return self._fail(StoreState.PARTIAL, "missing_evidence")
        if fact.parent_fact_ids:
            placeholders = ",".join("?" for _ in fact.parent_fact_ids)
            parent_rows = self.store._connection.execute(
                "SELECT fact_id, subject_id FROM clinical_facts WHERE fact_id IN "
                f"({placeholders})",  # noqa: S608
                fact.parent_fact_ids,
            ).fetchall()
            if len(parent_rows) != len(fact.parent_fact_ids):
                return self._fail(StoreState.PARTIAL, "missing_parent_fact")
            if any(row["subject_id"] != fact.subject_id for row in parent_rows):
                return self._fail(StoreState.CONFLICT, "parent_subject_mismatch")
        result = self._insert_simple(
            table="clinical_facts",
            id_column="fact_id",
            record_id=fact.fact_id,
            payload_json=fact.to_json(),
            extra_columns=("subject_id", "fact_type"),
            extra_values=(fact.subject_id, fact.fact_type),
            value=fact,
        )
        if result.ok and result.created:
            self.store._connection.executemany(
                "INSERT INTO fact_evidence(fact_id, locator_id) VALUES (?, ?)",
                ((fact.fact_id, item) for item in fact.evidence_ids),
            )
            self.store._connection.executemany(
                "INSERT INTO fact_parents(fact_id, parent_fact_id) VALUES (?, ?)",
                ((fact.fact_id, item) for item in fact.parent_fact_ids),
            )
        return result

    def put_conflict(self, conflict: ConflictSet) -> StoreResult[ConflictSet]:
        """Persist a conflict after every referenced fact exists."""

        denied = self._denied("write", "conflict")
        if denied is not None:
            return denied
        rows = self.store._connection.execute(
            "SELECT fact_id, subject_id FROM clinical_facts WHERE fact_id IN "
            f"({','.join('?' for _ in conflict.fact_ids)})",  # noqa: S608
            conflict.fact_ids,
        ).fetchall()
        if len(rows) != len(conflict.fact_ids):
            return self._fail(StoreState.PARTIAL, "missing_conflict_fact")
        if any(row["subject_id"] != conflict.subject_id for row in rows):
            return self._fail(StoreState.CONFLICT, "conflict_subject_mismatch")
        result = self._insert_simple(
            table="conflict_sets",
            id_column="conflict_id",
            record_id=conflict.conflict_id,
            payload_json=conflict.to_json(),
            extra_columns=("subject_id",),
            extra_values=(conflict.subject_id,),
            value=conflict,
        )
        if result.ok and result.created:
            self.store._connection.executemany(
                "INSERT INTO conflict_facts(conflict_id, fact_id) VALUES (?, ?)",
                ((conflict.conflict_id, item) for item in conflict.fact_ids),
            )
        return result

    def put_resolution(
        self,
        resolution: ResolutionEvent,
    ) -> StoreResult[ResolutionEvent]:
        """Persist a resolution after conflict and supersession checks."""

        denied = self._denied("write", "resolution")
        if denied is not None:
            return denied
        conflict_row = self.store._connection.execute(
            "SELECT conflict_id, payload_json FROM conflict_sets WHERE conflict_id = ?",
            (resolution.conflict_id,),
        ).fetchone()
        if conflict_row is None:
            return self._fail(StoreState.PARTIAL, "missing_conflict")
        try:
            conflict = ConflictSet.from_json(
                self.store._decode_payload("conflict_sets", conflict_row)
            )
        except EncryptionError as exc:
            return self._fail(StoreState.FAILURE, exc.code)
        selected = set(resolution.selected_fact_ids + resolution.rejected_fact_ids)
        if not selected.issubset(conflict.fact_ids):
            return self._fail(StoreState.CONFLICT, "resolution_fact_mismatch")
        if resolution.supersedes_resolution_id is not None:
            parent = self.store._connection.execute(
                "SELECT conflict_id FROM resolution_events WHERE resolution_id = ?",
                (resolution.supersedes_resolution_id,),
            ).fetchone()
            if parent is None:
                return self._fail(StoreState.PARTIAL, "missing_resolution_parent")
            if parent["conflict_id"] != resolution.conflict_id:
                return self._fail(StoreState.CONFLICT, "resolution_parent_mismatch")
        return self._insert_simple(
            table="resolution_events",
            id_column="resolution_id",
            record_id=resolution.resolution_id,
            payload_json=resolution.to_json(),
            extra_columns=("conflict_id", "supersedes_resolution_id"),
            extra_values=(
                resolution.conflict_id,
                resolution.supersedes_resolution_id,
            ),
            value=resolution,
        )

    def put_dataset(self, snapshot: DatasetSnapshot) -> StoreResult[DatasetSnapshot]:
        """Persist a dataset snapshot after source dependencies exist."""

        denied = self._denied("write", "dataset")
        if denied is not None:
            return denied
        checks = (
            ("artifacts", "artifact_id", snapshot.source_artifact_ids),
            ("clinical_facts", "fact_id", snapshot.source_fact_ids),
            ("dataset_snapshots", "snapshot_id", snapshot.parent_snapshot_ids),
        )
        for table, column, identifiers in checks:
            if not all(self._exists(table, column, item) for item in identifiers):
                return self._fail(StoreState.PARTIAL, "missing_dataset_source")
        return self._insert_simple(
            table="dataset_snapshots",
            id_column="snapshot_id",
            record_id=snapshot.snapshot_id,
            payload_json=snapshot.to_json(),
            value=snapshot,
        )

    def put_canonical(
        self,
        record: CanonicalRecord,
    ) -> StoreResult[CanonicalRecordVersion]:
        """Append a canonical-record version or return its identical version."""

        denied = self._denied("write", "canonical")
        if denied is not None:
            return denied
        fact_row = self.store._connection.execute(
            "SELECT subject_id FROM clinical_facts WHERE fact_id = ?",
            (record.fact_id,),
        ).fetchone()
        if fact_row is None:
            return self._fail(StoreState.PARTIAL, "missing_canonical_fact")
        if fact_row["subject_id"] != record.subject_id:
            return self._fail(StoreState.CONFLICT, "canonical_subject_mismatch")
        payload_json = record.to_json()
        payload_hash = canonical_digest(record.to_dict())
        existing = self.store._connection.execute(
            """
            SELECT version, created_revision FROM canonical_record_versions
            WHERE canonical_id = ? AND payload_hash = ?
            """,
            (record.canonical_id, payload_hash),
        ).fetchone()
        if existing is not None:
            version = CanonicalRecordVersion(
                record=record,
                version=int(existing["version"]),
                revision=int(existing["created_revision"]),
            )
            return StoreResult.success(
                version, created=False, revision=version.revision
            )
        row = self.store._connection.execute(
            "SELECT COALESCE(MAX(version), 0) + 1 AS version "
            "FROM canonical_record_versions WHERE canonical_id = ?",
            (record.canonical_id,),
        ).fetchone()
        version_number = int(row["version"])
        try:
            stored_payload = self.store._encode_payload(
                "canonical_record_versions",
                (record.canonical_id, version_number),
                payload_json,
            )
        except EncryptionError as exc:
            return self._fail(StoreState.FAILURE, exc.code)
        self.store._connection.execute(
            """
            INSERT INTO canonical_record_versions(
                canonical_id, version, subject_id, fact_id, payload_hash,
                payload_json, created_revision
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.canonical_id,
                version_number,
                record.subject_id,
                record.fact_id,
                payload_hash,
                stored_payload,
                self.revision,
            ),
        )
        version = CanonicalRecordVersion(
            record=record,
            version=version_number,
            revision=self.revision,
        )
        self.mutated = True
        return StoreResult.success(version, created=True, revision=self.revision)

    def put_job(self, job: JobMetadata) -> StoreResult[JobMetadata]:
        """Append a PHI-free job-state version idempotently."""

        denied = self._denied("write", "job")
        if denied is not None:
            return denied
        payload_json = job.to_json()
        payload_hash = canonical_digest(job.to_dict())
        existing = self.store._connection.execute(
            """
            SELECT job_id, version, payload_json, created_revision
            FROM job_metadata_versions
            WHERE job_id = ? AND payload_hash = ?
            """,
            (job.job_id, payload_hash),
        ).fetchone()
        if existing is not None:
            try:
                stored_job = JobMetadata.from_dict(
                    json.loads(
                        self.store._decode_payload("job_metadata_versions", existing)
                    )
                )
            except EncryptionError as exc:
                return self._fail(StoreState.FAILURE, exc.code)
            except (ValueError, TypeError, json.JSONDecodeError):
                return self._fail(StoreState.FAILURE, "stored_payload_invalid")
            return StoreResult.success(
                stored_job,
                created=False,
                revision=int(existing["created_revision"]),
            )
        row = self.store._connection.execute(
            "SELECT COALESCE(MAX(version), 0) + 1 AS version "
            "FROM job_metadata_versions WHERE job_id = ?",
            (job.job_id,),
        ).fetchone()
        version_number = int(row["version"])
        try:
            stored_payload = self.store._encode_payload(
                "job_metadata_versions",
                (job.job_id, version_number),
                payload_json,
            )
        except EncryptionError as exc:
            return self._fail(StoreState.FAILURE, exc.code)
        self.store._connection.execute(
            """
            INSERT INTO job_metadata_versions(
                job_id, version, state, payload_hash, payload_json, created_revision
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                job.job_id,
                version_number,
                job.state,
                payload_hash,
                stored_payload,
                self.revision,
            ),
        )
        self.mutated = True
        return StoreResult.success(job, created=True, revision=self.revision)

    def _insert_simple(
        self,
        *,
        table: str,
        id_column: str,
        record_id: str,
        payload_json: str,
        value: T,
        extra_columns: tuple[str, ...] = (),
        extra_values: tuple[Any, ...] = (),
    ) -> StoreResult[T]:
        existing_id = self.store._connection.execute(
            f"SELECT {id_column}, payload_json, created_revision "  # noqa: S608
            f"FROM {table} WHERE {id_column} = ?",
            (record_id,),
        ).fetchone()
        parser = cast(Callable[[str], T], getattr(type(value), "from_json"))
        if existing_id is not None:
            return self._existing(existing_id, value, parser, table=table)
        payload = json.loads(payload_json)
        payload_hash = canonical_digest(payload)
        existing_hash = self.store._connection.execute(
            f"SELECT {id_column}, payload_json, created_revision "  # noqa: S608
            f"FROM {table} WHERE payload_hash = ?",
            (payload_hash,),
        ).fetchone()
        if existing_hash is not None:
            return self._existing(
                existing_hash,
                value,
                parser,
                table=table,
                require_equal=False,
            )
        try:
            stored_payload = self.store._encode_payload(
                table,
                (record_id,),
                payload_json,
            )
        except EncryptionError as exc:
            return self._fail(StoreState.FAILURE, exc.code)
        columns = (
            id_column,
            *extra_columns,
            "payload_hash",
            "payload_json",
            "created_revision",
        )
        placeholders = ",".join("?" for _ in columns)
        try:
            self.store._connection.execute(
                f"INSERT INTO {table}({','.join(columns)}) VALUES ({placeholders})",  # noqa: S608
                (record_id, *extra_values, payload_hash, stored_payload, self.revision),
            )
        except (sqlite3.IntegrityError, StoreConstraintError):
            return self._fail(StoreState.CONFLICT, "record_conflict")
        self.mutated = True
        return StoreResult.success(value, created=True, revision=self.revision)

    def _existing(
        self,
        row: sqlite3.Row,
        requested: T,
        parser: Callable[[str], T],
        *,
        table: str,
        require_equal: bool = True,
    ) -> StoreResult[T]:
        try:
            existing = parser(self.store._decode_payload(table, row))
        except EncryptionError as exc:
            return self._fail(StoreState.FAILURE, exc.code)
        except (ValueError, TypeError, json.JSONDecodeError):
            return self._fail(StoreState.FAILURE, "stored_payload_invalid")
        if require_equal and existing != requested:
            return self._fail(StoreState.CONFLICT, "record_conflict")
        revision = (
            int(row["created_revision"]) if "created_revision" in row.keys() else None
        )
        return StoreResult.success(existing, created=False, revision=revision)

    def _exists(self, table: str, column: str, value: str) -> bool:
        return (
            self.store._connection.execute(
                f"SELECT 1 FROM {table} WHERE {column} = ?",  # noqa: S608
                (value,),
            ).fetchone()
            is not None
        )

    def _denied(self, operation: str, record_type: str) -> StoreResult[Any] | None:
        denied = self.store._denied(operation, record_type)
        if denied is not None:
            self.failed = True
        return denied

    def _fail(self, state: StoreState, code: str) -> StoreResult[Any]:
        self.failed = True
        return StoreResult.outcome(state, code, revision=self.revision)


class LocalJourneyStore:
    """Facade joining verified blobs to one atomic SQLite metadata graph.

    Args:
        root: Directory holding ``artifacts/`` and ``journey.sqlite3``.
        policy: Optional storage policy shared by both stores.
        encryption: Required encryption configuration shared by both stores.

    Raises:
        JourneyEncryptionError: If the configuration is missing, refused by
            the active profile, or incompatible with the persisted state.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        policy: StoragePolicy | None = None,
        encryption: JourneyStoreEncryption | None = None,
    ) -> None:
        root_path = Path(root)
        self.artifacts = LocalArtifactStore(
            root_path / "artifacts",
            policy=policy,
            encryption=encryption,
        )
        self.metadata = SQLiteJourneyStore(
            root_path / "journey.sqlite3",
            policy=policy,
            encryption=encryption,
        )

    @classmethod
    def migrate_to_encrypted(
        cls,
        root: str | Path,
        *,
        encryption: JourneyStoreEncryption,
        policy: StoragePolicy | None = None,
        batch_size: int = 256,
    ) -> StoreResult[LocalJourneyReencryptionReport]:
        """Encrypt an existing plaintext local Journey store in place.

        Artifacts are sealed first, then metadata. Both halves are resumable:
        call again after an interruption and only the remaining plaintext is
        sealed. Back up the store (closed, including its write-ahead log)
        before migrating and protect or destroy plaintext backups afterwards;
        migration cannot reach copies outside the store directory.

        Args:
            root: Directory holding ``artifacts/`` and ``journey.sqlite3``.
            encryption: Encrypted target configuration.
            policy: Optional storage policy.
            batch_size: Metadata rows per committed batch.

        Returns:
            Value-free reports for both halves, or a typed failure.
        """

        root_path = Path(root)
        artifacts = LocalArtifactStore.migrate_to_encrypted(
            root_path / "artifacts",
            encryption=encryption,
            policy=policy,
        )
        if not artifacts.ok or artifacts.value is None:
            return StoreResult.outcome(
                artifacts.state, artifacts.code or "artifact_migration_failed"
            )
        metadata = SQLiteJourneyStore.migrate_to_encrypted(
            root_path / "journey.sqlite3",
            encryption=encryption,
            policy=policy,
            batch_size=batch_size,
        )
        if not metadata.ok or metadata.value is None:
            return StoreResult.outcome(
                metadata.state, metadata.code or "metadata_migration_failed"
            )
        return StoreResult.success(
            LocalJourneyReencryptionReport(
                artifacts=artifacts.value,
                metadata=metadata.value,
            )
        )

    @property
    def encryption_state(self) -> str:
        """Return the metadata store's ``plaintext``/``migrating``/``encrypted``."""

        return self.metadata.encryption_state

    def reencrypt(
        self,
        *,
        batch_size: int = 256,
    ) -> StoreResult[LocalJourneyReencryptionReport]:
        """Re-seal artifacts and metadata under the provider's active key.

        Args:
            batch_size: Metadata rows per committed batch.

        Returns:
            Value-free reports for both halves, or a typed failure.
        """

        artifacts = self.artifacts.reencrypt()
        if not artifacts.ok or artifacts.value is None:
            return StoreResult.outcome(
                artifacts.state, artifacts.code or "artifact_reencrypt_failed"
            )
        metadata = self.metadata.reencrypt(batch_size=batch_size)
        if not metadata.ok or metadata.value is None:
            return StoreResult.outcome(
                metadata.state, metadata.code or "reencrypt_failed"
            )
        return StoreResult.success(
            LocalJourneyReencryptionReport(
                artifacts=artifacts.value,
                metadata=metadata.value,
            )
        )

    def close(self) -> None:
        """Close the metadata store."""

        self.metadata.close()

    def __enter__(self) -> "LocalJourneyStore":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def ingest_graph(
        self,
        artifact: ClinicalArtifact,
        content: bytes,
        *,
        evidence: Iterable[EvidenceLocator] = (),
        facts: Iterable[ClinicalFact] = (),
        conflicts: Iterable[ConflictSet] = (),
        committed_at: str,
    ) -> StoreResult[IngestedGraph]:
        """Atomically persist one verified source-to-fact metadata graph."""

        locators = tuple(evidence)
        fact_records = tuple(facts)
        conflict_records = tuple(conflicts)
        blob = self.artifacts.put_bytes(artifact, content)
        if not blob.ok:
            return StoreResult.outcome(blob.state, blob.code or "artifact_write_failed")

        failure: StoreResult[Any] | None = None
        created_metadata = False
        try:
            with self.metadata.transaction(committed_at=committed_at) as transaction:
                artifact_result = transaction.put_artifact(artifact)
                if not artifact_result.ok:
                    failure = artifact_result
                else:
                    created_metadata = created_metadata or artifact_result.created
                for locator in locators:
                    if failure is not None:
                        break
                    evidence_result = transaction.put_evidence(locator)
                    if not evidence_result.ok:
                        failure = evidence_result
                    else:
                        created_metadata = created_metadata or evidence_result.created
                for fact in fact_records:
                    if failure is not None:
                        break
                    fact_result = transaction.put_fact(fact)
                    if not fact_result.ok:
                        failure = fact_result
                    else:
                        created_metadata = created_metadata or fact_result.created
                for conflict in conflict_records:
                    if failure is not None:
                        break
                    conflict_result = transaction.put_conflict(conflict)
                    if not conflict_result.ok:
                        failure = conflict_result
                    else:
                        created_metadata = created_metadata or conflict_result.created
        except EncryptionError as exc:
            failure = StoreResult.outcome(StoreState.FAILURE, exc.code)
        except (LocalStoreError, sqlite3.Error, TypeError, ValueError):
            failure = StoreResult.outcome(StoreState.FAILURE, "transaction_failed")

        if failure is not None:
            if blob.created:
                self.artifacts.discard_if_created(artifact.content_hash)
            return StoreResult.outcome(
                failure.state, failure.code or "transaction_failed"
            )
        graph = IngestedGraph(
            artifact=artifact,
            evidence_count=len(locators),
            fact_count=len(fact_records),
            conflict_count=len(conflict_records),
        )
        return StoreResult.success(
            graph,
            created=blob.created or created_metadata,
            revision=(
                transaction.revision
                if created_metadata
                else self.metadata.latest_revision
            ),
        )


def _valid_timestamp(value: Any) -> bool:
    if not isinstance(value, str) or _TIMESTAMP_RE.fullmatch(value) is None:
        return False
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _valid_opaque_id(value: Any) -> bool:
    return isinstance(value, str) and _OPAQUE_ID_RE.fullmatch(value) is not None


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        return
    finally:
        os.close(descriptor)


def _record_identity(table: str, row: Any) -> tuple[str | int, ...]:
    return tuple(row[column] for column in PAYLOAD_RECORD_KEYS[table])


def _require_positive_bound(value: Any, name: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


class _PayloadHashMismatch(Exception):
    """Internal signal that a stored payload no longer matches its digest."""


@dataclass(slots=True)
class _Progress:
    examined: int = 0
    reencrypted: int = 0
    migrated: int = 0
    complete: bool = True

    @property
    def rewritten(self) -> int:
        return self.reencrypted + self.migrated

    def count(self, *, migrated: bool) -> None:
        if migrated:
            self.migrated += 1
        else:
            self.reencrypted += 1

    def report(self, state: str) -> ReencryptionReport:
        return ReencryptionReport(
            examined=self.examined,
            reencrypted=self.reencrypted,
            migrated=self.migrated,
            complete=self.complete,
            state=state,
        )

    def failure(self, code: str, state: str) -> StoreResult[ReencryptionReport]:
        self.complete = False
        return StoreResult.outcome(StoreState.FAILURE, code, value=self.report(state))


def _replace_file(path: Path, content: bytes) -> None:
    """Atomically replace ``path`` with owner-only ``content``."""

    descriptor, temp_name = tempfile.mkstemp(prefix=_BLOB_TEMP_PREFIX, dir=path.parent)
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        Path(temp_name).unlink(missing_ok=True)
        raise
    _fsync_directory(path.parent)


class _SQLiteKeyVerifiers:
    """Key verifiers persisted in the SQLite ``store_encryption_keys`` table."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def load_key_verifier(self, key_id: str) -> str | None:
        row = self._connection.execute(
            "SELECT verifier FROM store_encryption_keys WHERE key_id = ?",
            (key_id,),
        ).fetchone()
        return None if row is None else str(row["verifier"])

    def save_key_verifier(self, key_id: str, verifier: str) -> None:
        self._connection.execute(
            "INSERT OR IGNORE INTO store_encryption_keys(key_id, verifier) "
            "VALUES (?, ?)",
            (key_id, verifier),
        )


class _MarkerKeyVerifiers:
    """Key verifiers persisted in the artifact store's encryption marker."""

    def __init__(self, store: LocalArtifactStore) -> None:
        self._store = store

    def load_key_verifier(self, key_id: str) -> str | None:
        return self._store._key_verifiers.get(key_id)

    def save_key_verifier(self, key_id: str, verifier: str) -> None:
        self._store._save_key_verifier(key_id, verifier)
