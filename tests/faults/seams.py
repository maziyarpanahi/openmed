"""Test-only injection seams at documented service adapter boundaries.

Every seam is inert until its :class:`FaultInjector` is armed, counts each
pass through its named injection point (the negative control) and fires the
scheduled fault only on the resolved call (the positive control). Seams patch
process-level primitives only for paths under the case's own scope and restore
them on exit. Platform-specific mechanisms (errno values, SQLite error text,
HTTP transport errors) stay inside these adapters; nothing here reads payloads
into evidence beyond digests and counts.
"""

from __future__ import annotations

import asyncio
import errno
import os
import shutil
import sqlite3
import tempfile
import threading
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx

from .guard import (
    ALREADY_ARMED_CODE,
    NOT_ARMED_CODE,
    FaultInjectionForbiddenError,
    require_arm_permitted,
)
from .plan import FaultCase, FaultPlanError, ResolvedCase, sha256_digest

DISK_READ = "disk.read"
DISK_WRITE = "disk.write"
DISK_FSYNC = "disk.fsync"
DISK_REPLACE = "disk.replace"
DISK_LINK = "disk.link"
CLOCK_WALL = "clock.wall"
CLOCK_MONOTONIC = "clock.monotonic"
NETWORK_REQUEST = "network.request"
NETWORK_EGRESS = "network.egress"
SQLITE_COMMIT = "sqlite.commit"
STATE_RESTART = "state.restart"

_ERRNO_BY_FAULT = {
    "enospc": errno.ENOSPC,
    "eio": errno.EIO,
    "fsync_failure": errno.EIO,
    "partial_write": errno.ENOSPC,
}
CORRUPTION_FAULTS = frozenset({"truncated_state", "garbled_state"})


class SimulatedProcessKill(BaseException):
    """Abort the in-process run as SIGKILL would, after a crash image is taken.

    It derives from ``BaseException`` so ordinary ``except Exception`` cleanup
    does not intercept it. The durable view of the kill instant is the crash
    image, never the in-process state left behind by unwinding.
    """


class FaultInjector:
    """Seeded injector for one resolved case, in control or fault mode."""

    def __init__(self, resolved: ResolvedCase, *, mode: str) -> None:
        """Create an unarmed injector.

        Args:
            resolved: The case bound to its plan seed.
            mode: ``"control"`` counts seam passes without firing; ``"fault"``
                also fires the scheduled fault.
        """
        if mode not in {"control", "fault"}:
            raise FaultPlanError("fault_mode_invalid")
        self.resolved = resolved
        self.mode = mode
        self._hits: Counter[str] = Counter()
        self._fired = 0
        self._armed = False
        self._lock = threading.Lock()

    @property
    def case(self) -> FaultCase:
        """Return the declared case."""
        return self.resolved.case

    @property
    def is_armed(self) -> bool:
        """Return whether seams bound to this injector may act."""
        return self._armed

    @property
    def fired(self) -> int:
        """Return how many times the scheduled fault fired."""
        return self._fired

    def hits(self, point: str | None = None) -> int:
        """Return seam passes at ``point`` (default: the case's point)."""
        return self._hits[point or self.case.injection_point]

    @contextmanager
    def armed(self) -> Iterator["FaultInjector"]:
        """Arm the injector for one bounded scope.

        Raises:
            FaultInjectionForbiddenError: Outside the ``test`` profile or when
                already armed.
        """
        require_arm_permitted()
        if self._armed:
            raise FaultInjectionForbiddenError(ALREADY_ARMED_CODE)
        self._armed = True
        try:
            yield self
        finally:
            self._armed = False

    def hit(self, point: str) -> bool:
        """Count one pass through ``point`` and report whether to fire."""
        if not self._armed:
            raise FaultInjectionForbiddenError(NOT_ARMED_CODE)
        with self._lock:
            self._hits[point] += 1
            if self.mode != "fault" or point != self.case.injection_point:
                return False
            if not self.resolved.fires_on(self._hits[point]):
                return False
            self._fired += 1
            return True


class FaultSession:
    """Bind an injector to one working directory and its crash image."""

    def __init__(self, injector: FaultInjector, workdir: Path) -> None:
        """Create the case state directory under ``workdir``."""
        self.injector = injector
        self.state_dir = workdir / "state"
        self.crash_dir = workdir / "crash"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.crashed = False
        self._suspended = 0

    @property
    def case(self) -> FaultCase:
        """Return the declared case."""
        return self.injector.case

    @property
    def resolved(self) -> ResolvedCase:
        """Return the resolved case."""
        return self.injector.resolved

    @property
    def active(self) -> bool:
        """Return whether seams count and fire right now."""
        return self.injector.is_armed and self._suspended == 0

    @contextmanager
    def suspended(self) -> Iterator[None]:
        """Pass seams through without counting (setup and observation)."""
        self._suspended += 1
        try:
            yield
        finally:
            self._suspended -= 1

    def trip(self, point: str) -> bool:
        """Count one pass through ``point``; return whether the fault fires."""
        if not self.active:
            return False
        return self.injector.hit(point)

    def capture_crash_image(self) -> None:
        """Copy the durable state as it is at this instant (SQLite ``-shm`` aside)."""
        if self.crashed:
            raise FaultPlanError("crash_image_already_captured")
        with self.suspended():
            shutil.copytree(
                self.state_dir,
                self.crash_dir,
                ignore=shutil.ignore_patterns("*-shm"),
            )
        self.crashed = True

    def apply_fault(
        self,
        point: str,
        real: Callable[..., Any] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Apply the fired fault around an optional real call.

        ``after_commit`` kills and cancellations run ``real`` first; every other
        phase replaces it.
        """
        fault = self.case.fault
        after = self.case.phase == "after_commit"
        if fault == "process_kill":
            if after and real is not None:
                real(*args, **kwargs)
            self.capture_crash_image()
            raise SimulatedProcessKill(point)
        if fault == "cancellation":
            if after and real is not None:
                real(*args, **kwargs)
            raise asyncio.CancelledError()
        if fault in _ERRNO_BY_FAULT:
            raise OSError(_ERRNO_BY_FAULT[fault], fault)
        raise FaultPlanError("fault_unsupported_at_point")

    def call(
        self,
        point: str,
        real: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Route one real call through ``point``."""
        if not self.trip(point):
            return real(*args, **kwargs)
        return self.apply_fault(point, real, *args, **kwargs)

    def checkpoint(self, point: str, *, target: Path | None = None) -> bool:
        """Pass a named state transition with no wrapped call.

        Corruption faults damage ``target``; kills and cancellations raise.
        Other fired faults are returned as ``True`` for the caller to apply.
        """
        if not self.trip(point):
            return False
        if self.case.fault in CORRUPTION_FAULTS:
            if target is None:
                raise FaultPlanError("fault_target_missing")
            with self.suspended():
                corrupt_file(target, self.case.fault, self.resolved.rng("corruption"))
            return True
        if self.case.fault in {"process_kill", "cancellation"}:
            self.apply_fault(point)
        return True


@contextmanager
def _patched(patches: Sequence[tuple[Any, str, Any]]) -> Iterator[None]:
    originals: list[tuple[Any, str, Any]] = []
    try:
        for owner, name, value in patches:
            originals.append((owner, name, getattr(owner, name)))
            setattr(owner, name, value)
        yield
    finally:
        for owner, name, value in reversed(originals):
            setattr(owner, name, value)


def _require_armed(session: FaultSession) -> None:
    if not session.injector.is_armed:
        raise FaultInjectionForbiddenError(NOT_ARMED_CODE)


class _FaultingFile:
    """File proxy exposing ``write`` as the ``disk.write`` injection point."""

    def __init__(
        self,
        inner: Any,
        session: FaultSession,
        on_close: Callable[[], None],
    ) -> None:
        self._inner = inner
        self._session = session
        self._on_close = on_close

    def write(self, data: Any) -> Any:
        if not self._session.trip(DISK_WRITE):
            return self._inner.write(data)
        if self._session.case.fault == "partial_write":
            prefix = data[: len(data) // 2]
            if prefix:
                self._inner.write(prefix)
                self._inner.flush()
            raise OSError(errno.ENOSPC, "partial_write")
        return self._session.apply_fault(DISK_WRITE, self._inner.write, data)

    def close(self) -> None:
        self._on_close()
        self._inner.close()

    def __enter__(self) -> "_FaultingFile":
        self._inner.__enter__()
        return self

    def __exit__(self, *exc_info: Any) -> Any:
        self._on_close()
        return self._inner.__exit__(*exc_info)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


@contextmanager
def disk_faults(session: FaultSession, scope: Path) -> Iterator[None]:
    """Expose read, write, fsync, replace and link under ``scope`` as seams.

    Only paths (and descriptors opened for paths) under ``scope`` are counted
    or faulted; everything else passes through untouched.

    Args:
        session: The armed fault session.
        scope: Directory whose files the seam governs.
    """
    _require_armed(session)
    scope_real = os.path.realpath(scope)
    tracked: set[int] = set()
    real_replace = os.replace
    real_link = os.link
    real_fsync = os.fsync
    real_fdopen = os.fdopen
    real_mkstemp = tempfile.mkstemp
    real_named = tempfile.NamedTemporaryFile
    real_read_text = Path.read_text
    real_read_bytes = Path.read_bytes

    def in_scope(path: Any) -> bool:
        try:
            candidate = os.path.realpath(os.fspath(path))
        except TypeError:
            return False
        return candidate == scope_real or candidate.startswith(scope_real + os.sep)

    def replace(src: Any, dst: Any, *args: Any, **kwargs: Any) -> None:
        if session.active and in_scope(dst):
            return session.call(DISK_REPLACE, real_replace, src, dst, *args, **kwargs)
        return real_replace(src, dst, *args, **kwargs)

    def link(src: Any, dst: Any, *args: Any, **kwargs: Any) -> None:
        if session.active and in_scope(dst):
            return session.call(DISK_LINK, real_link, src, dst, *args, **kwargs)
        return real_link(src, dst, *args, **kwargs)

    def fsync(fd: Any) -> None:
        if session.active and fd in tracked:
            return session.call(DISK_FSYNC, real_fsync, fd)
        return real_fsync(fd)

    def mkstemp(*args: Any, **kwargs: Any) -> tuple[int, str]:
        descriptor, name = real_mkstemp(*args, **kwargs)
        if in_scope(name):
            tracked.add(descriptor)
        return descriptor, name

    def fdopen(fd: int, *args: Any, **kwargs: Any) -> Any:
        handle = real_fdopen(fd, *args, **kwargs)
        if fd in tracked:
            return _FaultingFile(handle, session, lambda: tracked.discard(fd))
        return handle

    def named_temporary_file(*args: Any, **kwargs: Any) -> Any:
        handle = real_named(*args, **kwargs)
        if in_scope(handle.name):
            descriptor = handle.fileno()
            tracked.add(descriptor)
            return _FaultingFile(handle, session, lambda: tracked.discard(descriptor))
        return handle

    def read_text(self: Path, *args: Any, **kwargs: Any) -> str:
        if session.active and in_scope(self):
            return session.call(DISK_READ, real_read_text, self, *args, **kwargs)
        return real_read_text(self, *args, **kwargs)

    def read_bytes(self: Path) -> bytes:
        if session.active and in_scope(self):
            return session.call(DISK_READ, real_read_bytes, self)
        return real_read_bytes(self)

    with _patched(
        (
            (os, "replace", replace),
            (os, "link", link),
            (os, "fsync", fsync),
            (os, "fdopen", fdopen),
            (tempfile, "mkstemp", mkstemp),
            (tempfile, "NamedTemporaryFile", named_temporary_file),
            (Path, "read_text", read_text),
            (Path, "read_bytes", read_bytes),
        )
    ):
        yield


class FaultClock:
    """Deterministic wall and monotonic clock with scheduled skew, jumps and stalls.

    Each read advances simulated time by ``tick_seconds``. The scheduled read
    applies ``clock_skew``/``clock_jump_forward`` (+seconds),
    ``clock_jump_backward`` (-seconds) or ``monotonic_stall`` (freeze) to the
    clock that was read, from that read on.
    """

    def __init__(
        self,
        session: FaultSession,
        *,
        wall_start: datetime,
        monotonic_start: float = 1000.0,
        tick_seconds: float = 1.0,
    ) -> None:
        """Create a clock bound to ``session``."""
        if wall_start.tzinfo is None:
            raise FaultPlanError("fault_clock_naive")
        self._session = session
        self._wall_start = wall_start.timestamp()
        self._monotonic_start = float(monotonic_start)
        self._tick = float(tick_seconds)
        self._reads = 0
        self._offsets = {CLOCK_WALL: 0.0, CLOCK_MONOTONIC: 0.0}
        self._frozen: dict[str, float] = {}

    def _read(self, point: str, start: float) -> float:
        self._reads += 1
        value = start + self._reads * self._tick + self._offsets[point]
        if self._session.trip(point):
            fault = self._session.case.fault
            seconds = abs(self._session.case.number("seconds", 3600.0))
            if fault in {"clock_skew", "clock_jump_forward"}:
                self._offsets[point] += seconds
            elif fault == "clock_jump_backward":
                self._offsets[point] -= seconds
            elif fault == "monotonic_stall":
                self._frozen.setdefault(point, value)
            else:
                self._session.apply_fault(point)
            value = start + self._reads * self._tick + self._offsets[point]
        return self._frozen.get(point, value)

    def wall_seconds(self) -> float:
        """Return simulated POSIX seconds."""
        return self._read(CLOCK_WALL, self._wall_start)

    def wall(self) -> datetime:
        """Return simulated aware UTC wall time."""
        return datetime.fromtimestamp(self.wall_seconds(), tz=timezone.utc)

    def monotonic(self) -> float:
        """Return simulated monotonic seconds."""
        return self._read(CLOCK_MONOTONIC, self._monotonic_start)


class FaultTransport(httpx.BaseTransport):
    """In-memory HTTP receiver exposing each request as ``network.request``.

    ``before_commit`` faults fail before the receiver sees the request;
    ``during_commit`` faults fail after the receiver accepted it, so the sender
    loses the response. The receiver keeps body and nonce digests only.
    """

    def __init__(self, session: FaultSession, *, status_code: int = 204) -> None:
        """Create a transport bound to ``session``."""
        _require_armed(session)
        self._session = session
        self._status_code = status_code
        self.received = 0
        self.body_digests: list[str] = []
        self.nonce_digests: list[str] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        fired = self._session.trip(NETWORK_REQUEST)
        fault = self._session.case.fault
        if fired and self._session.case.phase == "before_commit":
            self._raise(fault, request, sent=False)
        self.received += 1
        self.body_digests.append(sha256_digest(request.read()))
        nonce = request.headers.get("X-OpenMed-Nonce", "")
        self.nonce_digests.append(sha256_digest(nonce.encode("utf-8")))
        if fired:
            self._raise(fault, request, sent=True)
        return httpx.Response(self._status_code, request=request)

    def _raise(self, fault: str, request: httpx.Request, *, sent: bool) -> None:
        if fault == "connection_refused":
            raise httpx.ConnectError("connection_refused", request=request)
        if fault == "timeout":
            if sent:
                raise httpx.ReadTimeout("timeout", request=request)
            raise httpx.ConnectTimeout("timeout", request=request)
        if fault == "connection_reset":
            if sent:
                raise httpx.ReadError("connection_reset", request=request)
            raise httpx.ConnectError("connection_reset", request=request)
        self._session.apply_fault(NETWORK_REQUEST)


class LoopbackHTTPReceiver:
    """Loopback-only HTTP sink that records the digest of each received body."""

    def __init__(self) -> None:
        """Prepare an unstarted receiver bound to 127.0.0.1 on a free port."""
        self.body_digests: list[str] = []
        receiver = self

        class _Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - stdlib hook name
                length = int(self.headers.get("Content-Length", "0") or 0)
                receiver.body_digests.append(sha256_digest(self.rfile.read(length)))
                self.send_response(204)
                self.end_headers()

            def log_message(self, *_args: Any) -> None:
                return None

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="openmed-fault-receiver",
            daemon=True,
        )

    @property
    def received(self) -> int:
        """Return the number of bodies received."""
        return len(self.body_digests)

    @property
    def url(self) -> str:
        """Return the loopback URL of the sink."""
        port = self._server.server_address[1]
        return f"http://127.0.0.1:{port}/hooks/job"

    def __enter__(self) -> "LoopbackHTTPReceiver":
        self._thread.start()
        return self

    def __exit__(self, *_exc_info: Any) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


class SQLiteCommitSeam:
    """DB-API proxy exposing ``COMMIT`` as the ``sqlite.commit`` injection point."""

    def __init__(self, session: FaultSession, connection: Any) -> None:
        """Wrap ``connection`` for ``session``."""
        self._session = session
        self._connection = connection

    def execute(self, sql: Any, *args: Any) -> Any:
        if (
            isinstance(sql, str)
            and sql.strip().upper() == "COMMIT"
            and self._session.trip(SQLITE_COMMIT)
        ):
            if self._session.case.fault == "eio":
                raise sqlite3.OperationalError("disk I/O error")
            return self._session.apply_fault(
                SQLITE_COMMIT, self._connection.execute, sql, *args
            )
        return self._connection.execute(sql, *args)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._connection, name)


@contextmanager
def sqlite_commit_faults(session: FaultSession, store: Any) -> Iterator[None]:
    """Route ``store``'s SQLite connection through :class:`SQLiteCommitSeam`."""
    _require_armed(session)
    connection = store._connection  # noqa: SLF001 - test-only adapter boundary
    store._connection = SQLiteCommitSeam(session, connection)  # noqa: SLF001
    try:
        yield
    finally:
        store._connection = connection  # noqa: SLF001


def corrupt_file(path: Path, fault: str, rng: Any) -> tuple[str, str]:
    """Deterministically truncate or garble ``path``.

    Truncation keeps a seeded 25-75% prefix. Garbling XORs the first byte (so
    the format's magic or opening token always breaks) plus a seeded window.

    Args:
        path: File to damage in place.
        fault: ``truncated_state`` or ``garbled_state``.
        rng: Case RNG for the ``corruption`` purpose.

    Returns:
        Digests of the file before and after corruption.
    """
    original = path.read_bytes()
    if not original:
        raise FaultPlanError("fault_target_empty")
    if fault == "truncated_state":
        keep = int(len(original) * rng.uniform(0.25, 0.75))
        corrupted = original[: min(max(keep, 1), len(original) - 1)]
    elif fault == "garbled_state":
        data = bytearray(original)
        data[0] ^= 0xA5
        width = min(len(data), rng.randint(8, 64))
        offset = rng.randrange(0, len(data) - width + 1)
        for index in range(max(offset, 1), offset + width):
            data[index] ^= 0xA5
        corrupted = bytes(data)
    else:
        raise FaultPlanError("fault_unsupported_at_point")
    path.write_bytes(corrupted)
    return sha256_digest(original), sha256_digest(corrupted)


__all__ = [
    "CLOCK_MONOTONIC",
    "CLOCK_WALL",
    "CORRUPTION_FAULTS",
    "DISK_FSYNC",
    "DISK_LINK",
    "DISK_READ",
    "DISK_REPLACE",
    "DISK_WRITE",
    "FaultClock",
    "FaultInjector",
    "FaultSession",
    "FaultTransport",
    "LoopbackHTTPReceiver",
    "NETWORK_EGRESS",
    "NETWORK_REQUEST",
    "SQLITE_COMMIT",
    "SQLiteCommitSeam",
    "STATE_RESTART",
    "SimulatedProcessKill",
    "corrupt_file",
    "disk_faults",
    "sqlite_commit_faults",
]
