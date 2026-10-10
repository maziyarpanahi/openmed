"""Private, durable local replay protection shared by approval consumers."""

from __future__ import annotations

import math
import os
import sqlite3
import stat
from pathlib import Path

from .tokens import (
    ApprovalNonceStoreError,
    _validate_digest,
    _validate_timestamp,
)

_SCHEMA_VERSION = 1
_SCHEMA = """CREATE TABLE nonce_claims (
    nonce_digest TEXT PRIMARY KEY NOT NULL,
    expires_at INTEGER NOT NULL CHECK(typeof(expires_at) = 'integer')
) WITHOUT ROWID"""


class SQLiteApprovalNonceStore:
    """Atomically retain nonce digests across processes and restarts.

    Args:
        path: Database in an application-owned private local directory. All
            consumers must use the same file. Existing empty or invalid files
            are refused, never reset. Windows directory ACLs are caller-owned.
        timeout: Maximum SQLite lock wait in seconds, defaulting to five.

    Raises:
        ApprovalNonceStoreError: The file cannot be privately created, opened,
            validated, locked, or durably committed. Errors never include paths.
    """

    def __init__(self, path: str | os.PathLike[str], *, timeout: float = 5) -> None:
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout < 0
        ):
            raise ApprovalNonceStoreError("invalid_nonce_store_timeout", "nonce_store")
        self._timeout = timeout
        try:
            self._path = Path(path).absolute()
            created = False
            try:
                descriptor = os.open(
                    self._path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o600,
                )
            except FileExistsError:
                self._check_file()
            else:
                os.close(descriptor)
                created = True
            self._check_file(allow_empty=created)
            connection = self._connect()
            try:
                connection.execute("BEGIN EXCLUSIVE")
                if created:
                    connection.execute(_SCHEMA)
                    connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
                self._check_schema(connection)
                connection.commit()
            finally:
                connection.close()
        except (OSError, ValueError, TypeError, sqlite3.Error):
            raise ApprovalNonceStoreError(
                "nonce_store_unavailable", "nonce_store"
            ) from None

    def claim(self, nonce_digest: str, *, expires_at: int, now: int) -> bool:
        """Purge expired claims and durably claim a digest in one transaction.

        Args:
            nonce_digest: Canonical SHA-256 digest, never a raw nonce or token.
            expires_at: Exclusive Unix expiry as integer seconds.
            now: Trusted current Unix time as integer seconds.

        Returns:
            True for the first unexpired claim, False for a replay or expiry.

        Raises:
            ApprovalNonceStoreError: Storage fails; callers must deny dispatch.
        """

        _validate_digest(nonce_digest, "nonce_digest")
        _validate_timestamp(expires_at, "expires_at")
        _validate_timestamp(now, "now")
        try:
            self._check_file()
            connection = self._connect()
            try:
                connection.execute("BEGIN EXCLUSIVE")
                self._check_schema(connection)
                connection.execute(
                    "DELETE FROM nonce_claims WHERE expires_at <= ?", (now,)
                )
                claimed = False
                if expires_at > now:
                    cursor = connection.execute(
                        "INSERT INTO nonce_claims (nonce_digest, expires_at) VALUES (?, ?) "
                        "ON CONFLICT(nonce_digest) DO NOTHING",
                        (nonce_digest, expires_at),
                    )
                    claimed = cursor.rowcount == 1
                connection.commit()
                return claimed
            finally:
                connection.close()
        except (OSError, sqlite3.Error):
            raise ApprovalNonceStoreError(
                "nonce_store_unavailable", "nonce_store"
            ) from None

    def _check_file(self, *, allow_empty: bool = False) -> None:
        metadata = self._path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or (
            not allow_empty and metadata.st_size == 0
        ):
            raise ApprovalNonceStoreError("invalid_nonce_store_file", "nonce_store")
        if os.name == "posix":
            if metadata.st_uid != os.getuid():
                raise ApprovalNonceStoreError(
                    "invalid_nonce_store_owner", "nonce_store"
                )
            os.chmod(self._path, 0o600)

    def _connect(self) -> sqlite3.Connection:
        # mode=rw prevents SQLite from silently replacing a missing database.
        connection = sqlite3.connect(
            self._path.as_uri() + "?mode=rw",
            uri=True,
            timeout=self._timeout,
            isolation_level=None,
        )
        try:
            connection.execute("PRAGMA journal_mode = DELETE")
            # EXTRA also synchronizes the directory after deleting the journal.
            connection.execute("PRAGMA synchronous = EXTRA")
            return connection
        except sqlite3.Error:
            connection.close()
            raise

    @staticmethod
    def _check_schema(connection: sqlite3.Connection) -> None:
        if connection.execute("PRAGMA user_version").fetchone() != (_SCHEMA_VERSION,):
            raise ApprovalNonceStoreError(
                "unsupported_nonce_store_schema", "nonce_store"
            )
        objects = connection.execute(
            "SELECT type, name, sql FROM sqlite_master ORDER BY name"
        ).fetchall()
        if objects != [("table", "nonce_claims", _SCHEMA)]:
            raise ApprovalNonceStoreError("invalid_nonce_store_schema", "nonce_store")
        if connection.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
            raise ApprovalNonceStoreError("invalid_nonce_store_database", "nonce_store")

    def __repr__(self) -> str:
        """Return a representation without the private database path."""

        return "SQLiteApprovalNonceStore(<private>)"
