"""Bounded, content-free probes of untrusted media in disposable Linux workers.

This is crash/resource containment, not a complete OS sandbox or malware scan.
Only installed, closed-choice decoders run; no callback, filename or source text
crosses the public result boundary. Unsupported isolation fails closed.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import platform
import selectors
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class DecoderGuardError(ValueError):
    """Fixed input error that does not reflect supplied values."""


class DecoderKind(str, Enum):
    """Closed decoder choices; none accepts a user command or file path."""

    PDF = "pdf"
    TIFF = "tiff"
    DICOM = "dicom"
    WSI = "wsi_tiff"
    AUDIO = "wav"
    VIDEO = "video"


class DecodeStatus(str, Enum):
    """Stable terminal categories, independent of native exception text."""

    ACCEPTED = "probe_completed"
    MALFORMED = "malformed"
    LIMIT = "media_limit"
    MEMORY = "memory_limit"
    CPU = "cpu_limit"
    TIMEOUT = "timeout"
    CRASH = "decoder_crash"
    UNAVAILABLE = "backend_unavailable"
    ISOLATION = "isolation_unavailable"
    CANCELLED = "cancelled"
    FAILURE = "decoder_failure"


_EXIT = {
    DecodeStatus.ACCEPTED: 0,
    DecodeStatus.MALFORMED: 20,
    DecodeStatus.LIMIT: 21,
    DecodeStatus.MEMORY: 22,
    DecodeStatus.UNAVAILABLE: 23,
    DecodeStatus.ISOLATION: 24,
    DecodeStatus.FAILURE: 25,
}


def _require(condition: bool) -> None:
    if not condition:
        raise DecoderGuardError("invalid_decoder_guard_input")


@dataclass(frozen=True, slots=True, kw_only=True)
class DecoderLimits:
    """Explicit ceilings covering input, native allocations, work and geometry.

    Address-space bytes include the interpreter and imported native libraries.
    The wall deadline includes worker initialization and feeding, after process
    creation. OS process creation itself cannot be interrupted by this API.
    """

    input_bytes: int = 1_048_576
    address_space_bytes: int = 512 * 1024**2
    wall_seconds: float = 5.0
    cpu_seconds: int = 3
    pixels_per_frame: int = 4_000_000
    decoded_bytes: int = 32 * 1024**2
    frames: int = 8
    container_nodes: int = 1024

    def __post_init__(self) -> None:
        for value, low, high in (
            (self.input_bytes, 1, 64 * 1024**2),
            (self.address_space_bytes, 64 * 1024**2, 2 * 1024**3),
            (self.cpu_seconds, 1, 30),
            (self.pixels_per_frame, 1, 64_000_000),
            (self.decoded_bytes, 1, 256 * 1024**2),
            (self.frames, 1, 256),
            (self.container_nodes, 1, 10000),
        ):
            _require(type(value) is int and low <= value <= high)
        _require(
            type(self.wall_seconds) in (int, float)
            and math.isfinite(self.wall_seconds)
            and 0.05 <= self.wall_seconds <= 60
        )


@dataclass(frozen=True, slots=True)
class DecodeReport:
    """Opaque probe outcome; acceptance is not sanitization or clinical approval."""

    kind: DecoderKind
    status: DecodeStatus
    source_digest: str | None
    input_bytes: int
    limits_digest: str
    controls_applied: bool
    worker_reaped: bool

    @property
    def accepted(self) -> bool:
        """Whether the bounded decoder probe completed under verified controls."""
        return self.status is DecodeStatus.ACCEPTED and self.controls_applied

    def to_dict(self) -> dict[str, Any]:
        """Export fixed categories, hashes and counts without source content."""
        return {**asdict(self), "accepted": self.accepted, "clinical_readiness": False}


def isolation_supported() -> bool:
    """Return platform eligibility; the child must still install all controls."""
    return sys.platform == "linux" and platform.machine() in ("x86_64", "aarch64")


def probe_media(
    payload: bytes,
    kind: DecoderKind,
    *,
    limits: DecoderLimits = DecoderLimits(),
    cancelled: Callable[[], bool] = lambda: False,
) -> DecodeReport:
    """Probe bounded bytes using an isolated installed decoder.

    Args:
        payload: Caller-owned encoded bytes, never a filename or URL.
        kind: Closed local decoder choice.
        limits: Explicit input, address-space, CPU, wall and decode ceilings.
        cancelled: Fast local cancellation callback; errors cancel safely.

    Returns:
        Content-free result. Missing backends and unsupported isolation refuse
        execution. No decoded content is returned or written to temporary files.
    """
    return _execute(payload, kind, limits, cancelled=cancelled)


def _execute(
    payload: bytes,
    kind: DecoderKind,
    limits: DecoderLimits,
    *,
    cancelled: Callable[[], bool] = lambda: False,
    control: str = "",
) -> DecodeReport:
    _require(type(payload) is bytes and type(kind) is DecoderKind)
    _require(type(limits) is DecoderLimits and callable(cancelled))
    # Closed private fault controls are not inferred from media bytes.
    _require(
        control
        in (
            "",
            "crash",
            "timeout",
            "memory",
            "write",
            "network",
            "environment",
            "cpu",
            "log",
        )
    )
    declaration = json.dumps(asdict(limits), sort_keys=True, separators=(",", ":"))
    applied = False

    def result(status: DecodeStatus, reaped: bool = True) -> DecodeReport:
        return DecodeReport(
            kind,
            status,
            hashlib.sha256(payload).hexdigest()
            if len(payload) <= limits.input_bytes
            else None,
            len(payload),
            hashlib.sha256(declaration.encode()).hexdigest(),
            applied,
            reaped,
        )

    def stopped() -> bool:
        try:
            value = cancelled()
            return type(value) is not bool or value
        except Exception:
            return True

    if not payload or len(payload) > limits.input_bytes:
        return result(DecodeStatus.LIMIT)
    if stopped():
        return result(DecodeStatus.CANCELLED)
    if not isolation_supported():
        return result(DecodeStatus.ISOLATION)
    ready_read, ready_write = os.pipe()
    process = None
    status = DecodeStatus.FAILURE
    selector = selectors.DefaultSelector()
    try:
        # No inherited credentials, executable search overrides, Python path or
        # user cache locations. -I isolates Python; -B avoids bytecode writes.
        environment = {
            "PATH": os.defpath,
            "LC_ALL": "C.UTF-8",
            "HOME": "/nonexistent",
            "TMPDIR": "/nonexistent",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
        process = subprocess.Popen(
            [
                sys.executable,
                "-I",
                "-B",
                str(Path(__file__).resolve()),
                "--worker",
                kind.value,
                declaration,
                str(ready_write),
                control,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd="/",
            env=environment,
            start_new_session=True,
            close_fds=True,
            pass_fds=(ready_write,),
        )
        os.close(ready_write)
        ready_write = -1
        assert process.stdin is not None
        os.set_blocking(process.stdin.fileno(), False)
        os.set_blocking(ready_read, False)
        selector.register(process.stdin, selectors.EVENT_WRITE)
        offset = 0
        feed_failed = False
        deadline = time.monotonic() + limits.wall_seconds
        while process.poll() is None:
            try:
                applied |= os.read(ready_read, 1) == b"R"
            except BlockingIOError:
                pass
            if stopped():
                status = DecodeStatus.CANCELLED
                break
            if time.monotonic() >= deadline:
                status = DecodeStatus.TIMEOUT
                break
            for key, _ in selector.select(
                timeout=min(0.02, max(0, deadline - time.monotonic()))
            ):
                try:
                    offset += os.write(key.fd, payload[offset : offset + 65536])
                except (BlockingIOError, InterruptedError):
                    continue
                except OSError:
                    feed_failed = True
                    offset = len(payload)
                if offset == len(payload):
                    selector.unregister(process.stdin)
                    process.stdin.close()
        else:
            code = process.returncode
            status = next(
                (name for name, value in _EXIT.items() if value == code),
                DecodeStatus.FAILURE,
            )
            if code == -signal.SIGXCPU:
                status = DecodeStatus.CPU
            elif code is not None and code < 0:
                status = DecodeStatus.CRASH
        try:
            applied |= os.read(ready_read, 1) == b"R"
        except BlockingIOError:
            pass
        if status is DecodeStatus.ACCEPTED and feed_failed:
            status = DecodeStatus.FAILURE
        if status is DecodeStatus.ACCEPTED and not applied:
            status = DecodeStatus.ISOLATION
    except Exception:
        status = DecodeStatus.FAILURE
    finally:
        selector.close()
        if process is not None:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.wait()
            if process.stdin is not None:
                process.stdin.close()
        os.close(ready_read)
        if ready_write >= 0:
            os.close(ready_write)
    return result(status)


def _install_controls(limits: DecoderLimits) -> None:
    """Install irreversible Linux rlimits and a closed syscall filter in child."""
    import ctypes
    import resource

    _require(isolation_supported())
    for resource_id, ceiling in (
        (resource.RLIMIT_AS, limits.address_space_bytes),
        (resource.RLIMIT_CORE, 0),
        (resource.RLIMIT_FSIZE, 0),
        (resource.RLIMIT_NPROC, 0),
        (resource.RLIMIT_NOFILE, 64),
    ):
        resource.setrlimit(resource_id, (ceiling, ceiling))
        _require(resource.getrlimit(resource_id) == (ceiling, ceiling))

    resource.setrlimit(
        resource.RLIMIT_CPU, (limits.cpu_seconds, limits.cpu_seconds + 1)
    )

    _require(
        resource.getrlimit(resource.RLIMIT_CPU)
        == (limits.cpu_seconds, limits.cpu_seconds + 1)
    )

    class Filter(ctypes.Structure):
        _fields_ = [
            ("code", ctypes.c_ushort),
            ("jt", ctypes.c_ubyte),
            ("jf", ctypes.c_ubyte),
            ("k", ctypes.c_uint),
        ]

    class Program(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ushort), ("filters", ctypes.POINTER(Filter))]

    # Native 64-bit ABIs only. Unknown architectures and alternate x32 syscall
    # numbers are denied. No socket, clone, exec, path mutation or writable open.
    opens: tuple[tuple[int, int], ...]
    allowed: tuple[int, ...]
    if platform.machine() == "x86_64":
        arch, opens = 0xC000003E, ((2, 24), (257, 32))
        allowed = (
            0,
            1,
            3,
            4,
            5,
            6,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            19,
            20,
            21,
            24,
            25,
            28,
            32,
            33,
            35,
            39,
            60,
            63,
            72,
            78,
            79,
            89,
            96,
            97,
            98,
            99,
            102,
            104,
            107,
            108,
            110,
            131,
            158,
            186,
            202,
            204,
            217,
            218,
            228,
            229,
            230,
            231,
            262,
            267,
            269,
            273,
            292,
            318,
            332,
            334,
            436,
            439,
        )
    else:
        arch, opens = 0xC00000B7, ((56, 32),)
        allowed = (
            23,
            24,
            25,
            29,
            48,
            57,
            61,
            62,
            63,
            64,
            65,
            66,
            67,
            78,
            79,
            80,
            93,
            94,
            96,
            98,
            99,
            101,
            113,
            114,
            115,
            124,
            134,
            135,
            139,
            153,
            160,
            163,
            165,
            169,
            172,
            173,
            174,
            175,
            176,
            177,
            178,
            179,
            214,
            215,
            216,
            222,
            226,
            233,
            278,
            291,
            293,
            436,
            439,
        )
    deny, allow = 0x00050001, 0x7FFF0000  # EPERM / ALLOW
    instructions = [
        (0x20, 0, 0, 4),
        (0x15, 1, 0, arch),
        (0x06, 0, 0, 0x80000000),
        (0x20, 0, 0, 0),
    ]
    write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
    write_flags |= getattr(os, "O_TMPFILE", 0) & ~getattr(os, "O_DIRECTORY", 0)
    for syscall, flag_offset in opens:
        instructions.extend(
            [
                (0x15, 0, 4, syscall),
                (0x20, 0, 0, flag_offset),
                (0x45, 0, 1, write_flags),
                (0x06, 0, 0, deny),
                (0x06, 0, 0, allow),
                (0x20, 0, 0, 0),
            ]
        )
    for syscall in allowed:
        instructions.extend([(0x15, 0, 1, syscall), (0x06, 0, 0, allow)])
    instructions.append((0x06, 0, 0, deny))
    filters = (Filter * len(instructions))(*(Filter(*entry) for entry in instructions))
    program = Program(len(instructions), filters)
    libc = ctypes.CDLL(None, use_errno=True)
    _require(libc.prctl(4, 0, 0, 0, 0) == 0)  # PR_SET_DUMPABLE: no core-handler payload
    _require(libc.prctl(3, 0, 0, 0, 0) == 0)  # PR_GET_DUMPABLE
    _require(libc.prctl(38, 1, 0, 0, 0) == 0)  # PR_SET_NO_NEW_PRIVS
    _require(libc.prctl(22, 2, ctypes.byref(program), 0, 0) == 0)  # SECCOMP_MODE_FILTER


class _MediaLimit(Exception):
    pass


def _geometry(
    width: int, height: int, channels: int, frames: int, limits: DecoderLimits
) -> None:
    if min(width, height, channels, frames) <= 0:
        raise ValueError
    if (
        width * height > limits.pixels_per_frame
        or frames > limits.frames
        or width * height * channels * frames > limits.decoded_bytes
    ):
        raise _MediaLimit


def _tiff_directories(payload: bytes, limits: DecoderLimits) -> None:
    """Reject cyclic, out-of-bounds or excessive classic/BigTIFF directory graphs."""
    import struct

    if payload[:2] not in (b"II", b"MM"):
        raise ValueError
    endian = "<" if payload[:2] == b"II" else ">"

    def unpack(code: str, offset: int) -> int:
        return int(struct.unpack_from(endian + code, payload, offset)[0])

    version = unpack("H", 2)
    if version == 42:
        count_code, offset_code, count_size, entry_size, inline_size = (
            "H",
            "I",
            2,
            12,
            4,
        )
        pending = [unpack("I", 4)]
    elif version == 43 and unpack("H", 4) == 8 and unpack("H", 6) == 0:
        count_code, offset_code, count_size, entry_size, inline_size = (
            "Q",
            "Q",
            8,
            20,
            8,
        )
        pending = [unpack("Q", 8)]
    else:
        raise ValueError
    sizes = {
        1: 1,
        2: 1,
        3: 2,
        4: 4,
        5: 8,
        6: 1,
        7: 1,
        8: 2,
        9: 4,
        10: 8,
        11: 4,
        12: 8,
        13: 4,
        16: 8,
        17: 8,
        18: 8,
    }
    seen: set[int] = set()
    nodes = 0
    while pending:
        directory = pending.pop()
        if directory == 0:
            continue
        if directory in seen:
            raise ValueError
        seen.add(directory)
        count = unpack(count_code, directory)
        nodes += count + 1
        if nodes > limits.container_nodes:
            raise _MediaLimit
        end = directory + count_size + count * entry_size
        if end + inline_size > len(payload):
            raise ValueError
        for index in range(count):
            entry = directory + count_size + index * entry_size
            tag, typ = unpack("H", entry), unpack("H", entry + 2)
            quantity = unpack("I" if version == 42 else "Q", entry + 4)
            size = sizes.get(typ)
            if size is None:
                raise ValueError
            length = quantity * size
            value = entry + (8 if version == 42 else 12)
            offset = value if length <= inline_size else unpack(offset_code, value)
            if offset + length > len(payload):
                raise ValueError
            if tag in (330, 34665, 34853, 40965):
                if typ not in (4, 13, 16, 18):
                    raise ValueError
                if quantity + len(pending) > limits.container_nodes:
                    raise _MediaLimit
                pending.extend(
                    unpack("I" if size == 4 else "Q", offset + item * size)
                    for item in range(quantity)
                )
        pending.append(unpack(offset_code, end))


def _probe(payload: bytes, kind: DecoderKind, limits: DecoderLimits) -> None:
    if kind in (DecoderKind.TIFF, DecoderKind.WSI):
        _tiff_directories(payload, limits)
        from PIL import Image

        with Image.open(io.BytesIO(payload)) as image:
            if image.format != "TIFF":
                raise ValueError
            frame_count = int(getattr(image, "n_frames", 1))
            _geometry(image.width, image.height, 4, frame_count, limits)
            for frame in range(frame_count):
                image.seek(frame)
                _geometry(image.width, image.height, 4, frame_count, limits)
                with image.convert("RGBA") as decoded:
                    decoded.load()
    elif kind is DecoderKind.AUDIO:
        import wave

        with wave.open(io.BytesIO(payload), "rb") as audio:
            size = audio.getnframes() * audio.getnchannels() * audio.getsampwidth()
            if size > limits.decoded_bytes:
                raise _MediaLimit
            if audio.getcomptype() != "NONE" or not 1 <= audio.getnchannels() <= 8:
                raise ValueError
            if len(audio.readframes(audio.getnframes())) != size or size == 0:
                raise ValueError
    elif kind is DecoderKind.DICOM:
        import pydicom

        dataset = pydicom.dcmread(io.BytesIO(payload), force=False)
        if str(dataset.file_meta.TransferSyntaxUID) not in (
            "1.2.840.10008.1.2",
            "1.2.840.10008.1.2.1",
        ):
            raise ValueError
        frames = int(getattr(dataset, "NumberOfFrames", 1))
        channels = int(dataset.SamplesPerPixel) * math.ceil(
            int(dataset.BitsAllocated) / 8
        )
        _geometry(int(dataset.Columns), int(dataset.Rows), channels, frames, limits)
        if len(dataset) > limits.container_nodes:
            raise _MediaLimit
        dicom_pixels = dataset.pixel_array
        if dicom_pixels.nbytes > limits.decoded_bytes:
            raise _MediaLimit
    elif kind is DecoderKind.PDF:
        import pikepdf

        with pikepdf.open(
            io.BytesIO(payload), attempt_recovery=False, suppress_warnings=True
        ) as pdf:
            if len(pdf.pages) == 0:
                raise ValueError
            if (
                len(pdf.pages) > limits.frames
                or len(pdf.objects) > limits.container_nodes
            ):
                raise _MediaLimit
            if pdf.check_pdf_syntax():
                raise ValueError
            total = 0
            for obj in pdf.objects:
                if isinstance(obj, pikepdf.Stream):
                    total += len(obj.read_bytes())
                    if total > limits.decoded_bytes:
                        raise _MediaLimit
    else:
        import av

        with av.open(io.BytesIO(payload), mode="r") as container:
            if len(container.streams.video) != 1:
                raise ValueError
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 1
            _geometry(stream.width, stream.height, 3, 1, limits)
            count = 0
            for frame in container.decode(video=0):
                count += 1
                _geometry(frame.width, frame.height, 3, count, limits)
                frame.to_ndarray(format="rgb24")
            if count == 0:
                raise ValueError


def _fault_control(control: str, limits: DecoderLimits, payload: bytes) -> None:
    """Deterministic private child fault injection, never selected by media bytes."""
    if control == "crash":
        os.abort()
    elif control == "timeout":
        while True:
            time.sleep(1)
    elif control == "cpu":
        while True:
            pass
    elif control == "log":
        os.write(1, payload)
        os.write(2, payload)
    elif control == "memory":
        bytearray(limits.address_space_bytes * 2)
    elif control == "write":
        # Creation must be denied before any inode or raw payload can be written.
        try:
            fd = os.open(
                "/tmp/openmed-decoder-control", os.O_WRONLY | os.O_CREAT, 0o600
            )
        except PermissionError:
            return
        os.close(fd)
        raise RuntimeError
    elif control == "network":
        import socket

        try:
            socket.socket()
        except PermissionError:
            return
        raise RuntimeError
    elif control == "environment":
        if "OPENMED_HOSTILE_SENTINEL" in os.environ:
            raise RuntimeError


def _worker() -> int:
    try:
        _, marker, kind_name, declaration, ready, control = sys.argv
        _require(marker == "--worker")
        limits = DecoderLimits(**json.loads(declaration))
        kind = DecoderKind(kind_name)
        _install_controls(limits)
    except Exception:
        return _EXIT[DecodeStatus.ISOLATION]
    try:
        os.write(int(ready), b"R")
        os.close(int(ready))
        payload = sys.stdin.buffer.read(limits.input_bytes + 1)
        if not payload or len(payload) > limits.input_bytes:
            return _EXIT[DecodeStatus.LIMIT]
        if control:
            _fault_control(control, limits, payload)
        else:
            _probe(payload, kind, limits)
        return _EXIT[DecodeStatus.ACCEPTED]
    except MemoryError:
        return _EXIT[DecodeStatus.MEMORY]
    except _MediaLimit:
        return _EXIT[DecodeStatus.LIMIT]
    except ImportError:
        return _EXIT[DecodeStatus.UNAVAILABLE]
    except Exception:
        return _EXIT[DecodeStatus.MALFORMED]


if __name__ == "__main__":
    raise SystemExit(_worker())
