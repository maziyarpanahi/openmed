"""Private helpers for small, durable, owner-only service state files.

Used by the webhook signing keyring and the receiver replay ledger. Callers map
``OSError`` to their own controlled codes; nothing here formats paths or values
into messages.
"""

from __future__ import annotations

import os
import stat
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Optional


class StateFileTooLargeError(OSError):
    """Raised when a state file exceeds its configured size bound."""


def read_bounded(path: Path, max_bytes: int) -> Optional[bytes]:
    """Return a regular file's bytes, or ``None`` when it does not exist.

    Raises:
        StateFileTooLargeError: If the file is larger than ``max_bytes``.
        OSError: If the path is not a regular file or cannot be read.
    """
    try:
        with open(path, "rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise OSError("state path is not a regular file")
            data = handle.read(max_bytes + 1)
    except FileNotFoundError:
        return None
    if len(data) > max_bytes:
        raise StateFileTooLargeError("state file exceeds its size bound")
    return data


def atomic_write_private(path: Path, data: bytes) -> None:
    """Replace ``path`` with ``data`` atomically, readable only by the owner.

    The bytes go to a ``0600`` temporary file in the same directory, are
    flushed and fsynced, then moved over ``path`` with :func:`os.replace`. A
    crash at any point leaves either the previous file or the new one, never a
    partial file.

    Raises:
        OSError: If the directory or file cannot be written.
    """
    directory = path.parent
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(directory)
    )
    pending: Optional[str] = temp_name
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        pending = None
        _fsync_directory(directory)
    finally:
        if pending is not None:
            try:
                os.unlink(pending)
            except OSError:
                pass


@contextmanager
def exclusive_lock(lock_path: Path) -> Iterator[None]:
    """Hold an exclusive advisory lock on ``lock_path`` across processes.

    The lock file is created ``0600`` and never unlinked: removing it would
    split callers into separate lock domains. Closing the descriptor releases
    the lock, including when the holder crashes.

    Raises:
        OSError: If the lock file cannot be opened or locked.
    """
    lock_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(str(lock_path), flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError("lock path is not a regular file")
        if sys.platform == "win32":  # pragma: no cover - exercised on Windows only
            import msvcrt

            if os.fstat(descriptor).st_size == 0:
                os.write(descriptor, b"0")
            os.lseek(descriptor, 0, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":  # pragma: no cover - directories cannot be opened on Windows
        return
    try:
        descriptor = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)
