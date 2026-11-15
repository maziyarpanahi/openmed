#!/usr/bin/env python3
"""Gate a synthetic OpenMed service workflow under an OS-enforced no-egress boundary.

The gate runs everything below inside a Linux network namespace
(``unshare --user --map-root-user --net``) or a container started with
``--network none``. Only loopback exists inside the boundary.

* Workflow: the REST app (with GraphQL mounted) and the gRPC server start on
  loopback with a deterministic synthetic fixture loader in place of a cached
  checkpoint. The driver runs analyze, extract, and de-identify over REST,
  GraphQL, and gRPC, an async job, a webhook the in-process egress policy must
  refuse, client cancellation of a REST stream and a gRPC stream, Journey store
  persistence, a graceful restart that must preserve results and stored
  records, and a final graceful shutdown.
* Negative controls: IPv4 and IPv6 TCP, UDP, the system resolver, a raw DNS
  query, a child process, and raw libc sockets through ``ctypes`` each try to
  leave the host. Every one must fail with a "no route" class error. They run
  before ``openmed`` is imported, so no in-process guard can mask a hole.
* Positive controls: the same executors reach loopback listeners, which proves
  a failed negative control means "blocked" and not "broken probe".

Probe destinations are documentation addresses (RFC 5737, RFC 3849) and a
``.invalid`` name, so even a broken boundary contacts no real host.

The command writes a value-free attestation that validates against
``openmed/core/schemas/json/no_egress_attestation.schema.json`` and exits with:

* ``0`` when the workflow passed and every negative control was blocked;
* ``1`` when a control was not blocked, the boundary was not isolated, the
  workflow failed, or the in-boundary runner failed;
* ``2`` for invalid arguments or an unreadable fixture;
* ``3`` when the host has no supported boundary mechanism. This is a
  controlled skip and never a pass.

Examples::

    python scripts/security/no_egress_gate.py --output no-egress.json
    python scripts/security/no_egress_gate.py --boundary container \\
        --container-image openmed-service:local --output no-egress.json
    python scripts/security/no_egress_gate.py --verify no-egress.json
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import ctypes.util
import errno
import hashlib
import json
import os
import platform
import shutil
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[2]
if (ROOT / "openmed").is_dir() and str(ROOT) not in sys.path[:1]:
    # Test the checked-out sources, not whatever distribution is installed.
    sys.path.insert(0, str(ROOT))

DEFAULT_FIXTURE = ROOT / "tests/fixtures/security/no_egress_service_workflow.json"
IMAGE_ENV_VAR = "OPENMED_NO_EGRESS_IMAGE"
RESULT_MARKER = "OPENMED_NO_EGRESS_RESULT "

EXIT_PASSED = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_SKIPPED = 3

# Documentation-only destinations: never routed on the public internet.
PROBE_IPV4 = "192.0.2.10"
PROBE_IPV6 = "2001:db8::10"
PROBE_IPV4_RESOLVER = "192.0.2.53"
PROBE_DNS_NAME = "openmed-no-egress-probe.invalid"
PROBE_TCP_PORT = 443
PROBE_UDP_PORT = 9
PROBE_DNS_PORT = 53
PROBE_TIMEOUT_SECONDS = 3.0
SUBPROCESS_TIMEOUT_SECONDS = 30.0
RESOLVER_TIMEOUT_SECONDS = 20.0

SERVICE_START_TIMEOUT_SECONDS = 180.0
SERVICE_STOP_TIMEOUT_SECONDS = 60.0
REQUEST_TIMEOUT_SECONDS = 60.0
JOB_TIMEOUT_SECONDS = 120.0
DEFAULT_RUNNER_TIMEOUT_SECONDS = 900.0
CONTAINER_WORKSPACE = "/workspace"
CONTAINER_FIXTURE = "/no-egress/fixture.json"

_PROXY_ENV_VARS = frozenset(
    {
        "ALL_PROXY",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "NO_PROXY",
        "all_proxy",
        "https_proxy",
        "http_proxy",
        "no_proxy",
        "grpc_proxy",
    }
)
_SIOCGIFFLAGS = 0x8913
_SIOCSIFFLAGS = 0x8914
_IFF_UP = 0x1
_IFF_LOOPBACK = 0x8

_BLOCKED_ERRNO_REASONS: Mapping[int, str] = {
    errno.ENETUNREACH: "network_unreachable",
    errno.EADDRNOTAVAIL: "address_unavailable",
    errno.EAFNOSUPPORT: "family_unsupported",
    errno.EPROTONOSUPPORT: "family_unsupported",
    errno.EPERM: "permission_denied",
    errno.EACCES: "permission_denied",
}
_PEER_ERRNO_REASONS: Mapping[int, str] = {
    errno.ECONNREFUSED: "peer_refused",
    errno.ECONNRESET: "peer_reset",
}
_TIMEOUT_ERRNOS = frozenset(
    {errno.ETIMEDOUT, errno.EINPROGRESS, errno.EAGAIN, errno.EWOULDBLOCK}
)
_RESOLVER_UNREACHABLE = frozenset(
    code
    for code in (
        getattr(socket, "EAI_AGAIN", None),
        getattr(socket, "EAI_FAIL", None),
    )
    if code is not None
)
_RESOLVER_ANSWERED = frozenset(
    code
    for code in (
        getattr(socket, "EAI_NONAME", None),
        getattr(socket, "EAI_NODATA", None),
    )
    if code is not None
)

_CHILD_PROBE = r"""
import json, socket, sys
kind, family, host, port = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
try:
    if kind == "tcp":
        fam = socket.AF_INET6 if family == "ipv6" else socket.AF_INET
        sock = socket.socket(fam, socket.SOCK_STREAM)
        sock.settimeout(3.0)
        try:
            sock.connect((host, port))
        finally:
            sock.close()
        out = {"kind": "connected"}
    else:
        socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        out = {"kind": "resolved"}
except socket.gaierror as exc:
    out = {"kind": "gai_error", "code": exc.errno}
except TimeoutError:
    out = {"kind": "timeout"}
except OSError as exc:
    out = {"kind": "error", "code": exc.errno}
print(json.dumps(out))
"""


# ---------------------------------------------------------------------------
# Probes (stdlib only; run before openmed is imported)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbeOutcome:
    """Raw, value-free result of one probe attempt.

    Attributes:
        kind: ``connected``, ``sent``, ``replied``, ``resolved``, ``error``,
            ``gai_error``, ``timeout``, or ``unavailable``.
        code: The ``errno`` or ``EAI_*`` code for error outcomes.
    """

    kind: str
    code: int | None = None


_OUTCOME_KINDS = frozenset(
    {
        "connected",
        "sent",
        "replied",
        "resolved",
        "error",
        "gai_error",
        "timeout",
        "unavailable",
    }
)


def _error_reason(code: int | None) -> str:
    if code in _BLOCKED_ERRNO_REASONS:
        return _BLOCKED_ERRNO_REASONS[code]
    if code in _PEER_ERRNO_REASONS:
        return _PEER_ERRNO_REASONS[code]
    if code == errno.EHOSTUNREACH:
        return "host_unreachable"
    if code in _TIMEOUT_ERRNOS:
        return "timeout"
    return "probe_error"


def classify_negative(outcome: ProbeOutcome) -> tuple[str, str]:
    """Classify an external-egress probe as blocked or not.

    Only errors the kernel raises when no route or address exists count as
    blocked. A timeout or a host-unreachable answer is inconclusive, because
    packets may have left the host; both fail the gate.

    Args:
        outcome: The raw probe outcome.

    Returns:
        ``(status, reason_code)`` with status ``blocked``, ``not_blocked``,
        ``inconclusive``, or ``not_run``.
    """

    if outcome.kind == "connected":
        return "not_blocked", "connection_established"
    if outcome.kind == "sent":
        return "not_blocked", "datagram_sent"
    if outcome.kind == "replied":
        return "not_blocked", "resolver_answered"
    if outcome.kind == "resolved":
        return "not_blocked", "name_resolved"
    if outcome.kind == "timeout":
        return "inconclusive", "timeout"
    if outcome.kind == "unavailable":
        return "not_run", "executor_unavailable"
    if outcome.kind == "gai_error":
        if outcome.code in _RESOLVER_UNREACHABLE:
            return "blocked", "resolver_unreachable"
        if outcome.code in _RESOLVER_ANSWERED:
            return "not_blocked", "resolver_answered"
        return "inconclusive", "probe_error"
    if outcome.kind == "error":
        reason = _error_reason(outcome.code)
        if outcome.code in _BLOCKED_ERRNO_REASONS:
            return "blocked", reason
        if outcome.code in _PEER_ERRNO_REASONS:
            return "not_blocked", reason
        return "inconclusive", reason
    return "inconclusive", "probe_error"


def classify_positive(outcome: ProbeOutcome) -> tuple[str, str]:
    """Classify a loopback probe as reachable or not.

    Args:
        outcome: The raw probe outcome.

    Returns:
        ``(status, reason_code)`` with status ``reachable``, ``unreachable``,
        or ``not_run``.
    """

    if outcome.kind in {"connected", "replied", "resolved"}:
        return "reachable", "loopback_reachable"
    if outcome.kind == "unavailable":
        return "not_run", "executor_unavailable"
    if outcome.kind == "sent":
        return "unreachable", "timeout"
    if outcome.kind == "timeout":
        return "unreachable", "timeout"
    if outcome.kind == "gai_error":
        return "unreachable", "probe_error"
    return "unreachable", _error_reason(outcome.code)


def _family(name: str) -> socket.AddressFamily:
    return socket.AF_INET6 if name == "ipv6" else socket.AF_INET


def tcp_probe(family: str, host: str, port: int) -> ProbeOutcome:
    """Attempt one TCP connection with the Python socket module."""

    try:
        sock = socket.socket(_family(family), socket.SOCK_STREAM)
    except OSError as exc:
        return ProbeOutcome("error", exc.errno)
    sock.settimeout(PROBE_TIMEOUT_SECONDS)
    try:
        sock.connect((host, port))
    except TimeoutError:
        return ProbeOutcome("timeout")
    except OSError as exc:
        return ProbeOutcome("error", exc.errno)
    finally:
        sock.close()
    return ProbeOutcome("connected")


def udp_probe(
    family: str, host: str, port: int, payload: bytes, *, expect_reply: bool
) -> ProbeOutcome:
    """Send one UDP datagram and optionally wait for a reply."""

    try:
        sock = socket.socket(_family(family), socket.SOCK_DGRAM)
    except OSError as exc:
        return ProbeOutcome("error", exc.errno)
    sock.settimeout(PROBE_TIMEOUT_SECONDS)
    try:
        sock.sendto(payload, (host, port))
        if not expect_reply:
            return ProbeOutcome("sent")
        reply, _ = sock.recvfrom(4096)
    except TimeoutError:
        return ProbeOutcome("timeout")
    except OSError as exc:
        return ProbeOutcome("error", exc.errno)
    finally:
        sock.close()
    if len(reply) >= 12 and reply[:2] == payload[:2] and reply[2] & 0x80:
        return ProbeOutcome("replied")
    return ProbeOutcome("error", None)


def dns_query_packet(name: str, transaction_id: int) -> bytes:
    """Return a minimal DNS ``A`` query for ``name``."""

    header = struct.pack("!HHHHHH", transaction_id, 0x0100, 1, 0, 0, 0)
    labels = b"".join(
        bytes([len(label)]) + label.encode("ascii") for label in name.split(".")
    )
    return header + labels + b"\x00" + struct.pack("!HH", 1, 1)


def resolver_probe(host: str) -> ProbeOutcome:
    """Resolve ``host`` through the system resolver with a bounded wait."""

    holder: dict[str, ProbeOutcome] = {}

    def _resolve() -> None:
        try:
            socket.getaddrinfo(host, PROBE_TCP_PORT, type=socket.SOCK_STREAM)
        except socket.gaierror as exc:
            holder["outcome"] = ProbeOutcome("gai_error", exc.errno)
        except OSError as exc:
            holder["outcome"] = ProbeOutcome("error", exc.errno)
        else:
            holder["outcome"] = ProbeOutcome("resolved")

    worker = threading.Thread(target=_resolve, daemon=True)
    worker.start()
    worker.join(RESOLVER_TIMEOUT_SECONDS)
    return holder.get("outcome", ProbeOutcome("timeout"))


def subprocess_probe(kind: str, family: str, host: str, port: int) -> ProbeOutcome:
    """Run a probe in an isolated child interpreter that never imports OpenMed."""

    # A minimal environment: no proxies, no OpenMed flags, no in-process guard.
    env = {key: os.environ[key] for key in ("PATH", "SYSTEMROOT") if key in os.environ}
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                "-c",
                _CHILD_PROBE,
                kind,
                family,
                host,
                str(port),
            ],
            capture_output=True,
            env=env,
            text=True,
            timeout=SUBPROCESS_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return ProbeOutcome("timeout")
    except OSError:
        return ProbeOutcome("unavailable")
    lines = completed.stdout.strip().splitlines()
    if completed.returncode != 0 or not lines:
        return ProbeOutcome("unavailable")
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError:
        return ProbeOutcome("unavailable")
    kind_value = payload.get("kind") if isinstance(payload, dict) else None
    code = payload.get("code") if isinstance(payload, dict) else None
    if kind_value not in _OUTCOME_KINDS or (code is not None and type(code) is not int):
        return ProbeOutcome("unavailable")
    return ProbeOutcome(str(kind_value), code)


def sockaddr_bytes(family: socket.AddressFamily, host: str, port: int) -> bytes:
    """Return a native ``sockaddr_in``/``sockaddr_in6`` for libc calls."""

    packed = socket.inet_pton(family, host)
    bsd_layout = sys.platform.startswith(("darwin", "freebsd", "openbsd", "netbsd"))
    if family == socket.AF_INET:
        if bsd_layout:
            return struct.pack("!BBH4s8x", 16, int(family), port, packed)
        return struct.pack("=H", int(family)) + struct.pack("!H4s8x", port, packed)
    if bsd_layout:
        return struct.pack("!BBHI16s", 28, int(family), port, 0, packed) + struct.pack(
            "=I", 0
        )
    return (
        struct.pack("=H", int(family))
        + struct.pack("!HI16s", port, 0, packed)
        + struct.pack("=I", 0)
    )


def _libc() -> Any:
    name = ctypes.util.find_library("c")
    try:
        return ctypes.CDLL(name, use_errno=True)
    except OSError:
        return ctypes.CDLL(None, use_errno=True)


def native_probe(transport: str, family: str, host: str, port: int) -> ProbeOutcome:
    """Attempt one connection or datagram through raw libc calls via ctypes.

    This path never touches Python's ``socket`` module, so an in-process
    monkeypatch cannot intercept it; only the OS boundary can stop it.
    """

    try:
        libc = _libc()
        c_socket = libc.socket
        c_socket.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
        c_socket.restype = ctypes.c_int
        c_close = libc.close
        c_close.argtypes = [ctypes.c_int]
        c_setsockopt = libc.setsockopt
        c_setsockopt.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint32,
        ]
        c_connect = libc.connect
        c_connect.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
        c_connect.restype = ctypes.c_int
        c_sendto = libc.sendto
        c_sendto.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint32,
        ]
        c_sendto.restype = ctypes.c_ssize_t
    except (AttributeError, OSError):
        return ProbeOutcome("unavailable")

    address_family = _family(family)
    kind = socket.SOCK_STREAM if transport == "tcp" else socket.SOCK_DGRAM
    fd = c_socket(int(address_family), int(kind), 0)
    if fd < 0:
        return ProbeOutcome("error", ctypes.get_errno())
    try:
        timeval = struct.pack("@ll", int(PROBE_TIMEOUT_SECONDS), 0)
        c_setsockopt(
            fd,
            socket.SOL_SOCKET,
            getattr(socket, "SO_SNDTIMEO", 21),
            timeval,
            len(timeval),
        )
        address = sockaddr_bytes(address_family, host, port)
        if transport == "tcp":
            if c_connect(fd, address, len(address)) == 0:
                return ProbeOutcome("connected")
        else:
            payload = b"\x00"
            if c_sendto(fd, payload, len(payload), 0, address, len(address)) >= 0:
                return ProbeOutcome("sent")
        code = ctypes.get_errno()
        if code in _TIMEOUT_ERRNOS:
            return ProbeOutcome("timeout")
        return ProbeOutcome("error", code)
    finally:
        c_close(fd)


class LoopbackResponders:
    """TCP accept loop and UDP/DNS responder bound to IPv4 loopback."""

    def __init__(self) -> None:
        self.tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.tcp.bind(("127.0.0.1", 0))
        self.tcp.listen(16)
        self.udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp.bind(("127.0.0.1", 0))
        self.tcp_port = int(self.tcp.getsockname()[1])
        self.udp_port = int(self.udp.getsockname()[1])
        self._threads = [
            threading.Thread(target=self._accept, daemon=True),
            threading.Thread(target=self._respond, daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    def _accept(self) -> None:
        while True:
            try:
                connection, _ = self.tcp.accept()
            except OSError:
                return
            connection.close()

    def _respond(self) -> None:
        while True:
            try:
                data, peer = self.udp.recvfrom(4096)
            except OSError:
                return
            if not peer:
                return
            reply = bytearray(data)
            if len(reply) >= 3:
                reply[2] |= 0x80
            with contextlib.suppress(OSError):
                self.udp.sendto(bytes(reply), peer)

    def close(self) -> None:
        """Stop both responders."""

        for sock in (self.tcp, self.udp):
            with contextlib.suppress(OSError):
                sock.shutdown(socket.SHUT_RDWR)
            sock.close()


NegativeProbe = Callable[[], ProbeOutcome]


def negative_probes() -> dict[str, NegativeProbe]:
    """Return the external-egress negative controls keyed by case identifier."""

    query = dns_query_packet(PROBE_DNS_NAME, 0x0E9E)
    return {
        "ipv4_tcp_socket": lambda: tcp_probe("ipv4", PROBE_IPV4, PROBE_TCP_PORT),
        "ipv6_tcp_socket": lambda: tcp_probe("ipv6", PROBE_IPV6, PROBE_TCP_PORT),
        "ipv4_udp_socket": lambda: udp_probe(
            "ipv4", PROBE_IPV4, PROBE_UDP_PORT, b"\x00", expect_reply=False
        ),
        "ipv6_udp_socket": lambda: udp_probe(
            "ipv6", PROBE_IPV6, PROBE_UDP_PORT, b"\x00", expect_reply=False
        ),
        "dns_system_resolver": lambda: resolver_probe(PROBE_DNS_NAME),
        "dns_udp_query": lambda: udp_probe(
            "ipv4", PROBE_IPV4_RESOLVER, PROBE_DNS_PORT, query, expect_reply=False
        ),
        "subprocess_ipv4_tcp": lambda: subprocess_probe(
            "tcp", "ipv4", PROBE_IPV4, PROBE_TCP_PORT
        ),
        "subprocess_ipv6_tcp": lambda: subprocess_probe(
            "tcp", "ipv6", PROBE_IPV6, PROBE_TCP_PORT
        ),
        "subprocess_dns": lambda: subprocess_probe(
            "dns", "any", PROBE_DNS_NAME, PROBE_TCP_PORT
        ),
        "native_ipv4_tcp": lambda: native_probe(
            "tcp", "ipv4", PROBE_IPV4, PROBE_TCP_PORT
        ),
        "native_ipv6_tcp": lambda: native_probe(
            "tcp", "ipv6", PROBE_IPV6, PROBE_TCP_PORT
        ),
        "native_ipv4_udp": lambda: native_probe(
            "udp", "ipv4", PROBE_IPV4, PROBE_UDP_PORT
        ),
    }


def positive_probes(responders: LoopbackResponders) -> dict[str, NegativeProbe]:
    """Return the loopback positive controls keyed by case identifier."""

    query = dns_query_packet(PROBE_DNS_NAME, 0x10CA)
    datagram = b"\x5a\x5a" + b"\x00" * 10
    return {
        "loopback_ipv4_tcp_socket": lambda: tcp_probe(
            "ipv4", "127.0.0.1", responders.tcp_port
        ),
        "loopback_ipv4_udp_socket": lambda: udp_probe(
            "ipv4", "127.0.0.1", responders.udp_port, datagram, expect_reply=True
        ),
        "loopback_dns_udp_query": lambda: udp_probe(
            "ipv4", "127.0.0.1", responders.udp_port, query, expect_reply=True
        ),
        "loopback_system_resolver": lambda: resolver_probe("localhost"),
        "loopback_subprocess_ipv4_tcp": lambda: subprocess_probe(
            "tcp", "ipv4", "127.0.0.1", responders.tcp_port
        ),
        "loopback_native_ipv4_tcp": lambda: native_probe(
            "tcp", "ipv4", "127.0.0.1", responders.tcp_port
        ),
    }


def run_probe_set(
    probes: Mapping[str, NegativeProbe],
    classify: Callable[[ProbeOutcome], tuple[str, str]],
) -> list[dict[str, str]]:
    """Run probes and return value-free ``id``/``status``/``reason_code`` rows."""

    results: list[dict[str, str]] = []
    for case_id, probe in probes.items():
        try:
            outcome = probe()
        except Exception:  # a crashing probe proves nothing
            outcome = ProbeOutcome("unavailable")
        status, reason = classify(outcome)
        results.append({"id": case_id, "status": status, "reason_code": reason})
    return results


# ---------------------------------------------------------------------------
# Boundary setup and inspection (Linux only)
# ---------------------------------------------------------------------------


def _interface_flags(sock: socket.socket, name: str) -> int:
    import fcntl

    request = struct.pack("16sH14s", name.encode("ascii")[:15], 0, b"\x00" * 14)
    reply = fcntl.ioctl(sock.fileno(), _SIOCGIFFLAGS, request)
    return int(struct.unpack("16sH", reply[:18])[1])


def bring_loopback_up() -> bool:
    """Bring ``lo`` up in a fresh network namespace; return success."""

    import fcntl

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            flags = _interface_flags(sock, "lo")
            if not flags & _IFF_UP:
                request = struct.pack("16sH14s", b"lo", flags | _IFF_UP, b"\x00" * 14)
                fcntl.ioctl(sock.fileno(), _SIOCSIFFLAGS, request)
    except OSError:
        return False
    return True


def _count_routes(path: str, device_index: int, loopback: set[str]) -> int:
    try:
        lines = Path(path).read_text(encoding="ascii").splitlines()
    except FileNotFoundError:
        return 0
    except OSError:
        return 1  # unreadable: fail closed
    count = 0
    for line in lines:
        fields = line.split()
        if len(fields) <= abs(device_index) or fields[0] == "Iface":
            continue
        if fields[device_index] not in loopback:
            count += 1
    return count


def inspect_boundary() -> tuple[dict[str, int], dict[str, int]]:
    """Count interfaces and non-loopback routes visible inside the boundary.

    Interfaces whose flags cannot be read count as non-loopback and up, and an
    unreadable route table counts as one route, so inspection fails closed.
    """

    try:
        names = [name for _, name in socket.if_nameindex()]
    except OSError:
        names = []
    loopback_up = 0
    non_loopback_up = 0
    loopback_names: set[str] = set()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            for name in names:
                try:
                    flags = _interface_flags(sock, name)
                except (OSError, UnicodeEncodeError):
                    non_loopback_up += 1
                    continue
                if flags & _IFF_LOOPBACK:
                    loopback_names.add(name)
                if flags & _IFF_UP:
                    if flags & _IFF_LOOPBACK:
                        loopback_up += 1
                    else:
                        non_loopback_up += 1
    except OSError:
        non_loopback_up += 1
    interfaces = {
        "total": len(names),
        "loopback_up": loopback_up,
        "non_loopback_up": non_loopback_up,
    }
    routes = {
        "ipv4_non_loopback": _count_routes("/proc/net/route", 0, loopback_names),
        "ipv6_non_loopback": _count_routes("/proc/net/ipv6_route", -1, loopback_names),
    }
    return interfaces, routes


# ---------------------------------------------------------------------------
# Fixture and synthetic model
# ---------------------------------------------------------------------------


class FixtureError(ValueError):
    """Raised when the workflow fixture is missing or malformed."""


@dataclass(frozen=True)
class WorkflowFixture:
    """Synthetic workflow inputs; all values are synthetic and PHI-free."""

    model_name: str
    lexicon: tuple[tuple[str, str, bool], ...]
    document: str
    stream_repeat: int
    stream_chunk_size: int
    journey_recorded_at: str

    @property
    def expected_entities(self) -> list[list[Any]]:
        """Return sorted ``[label, start, end]`` lexicon matches."""

        return _lexicon_matches(self.document, self.lexicon)

    @property
    def phi_terms(self) -> tuple[str, ...]:
        """Return the lexicon terms that must never survive de-identification."""

        return tuple(term for term, _, phi in self.lexicon if phi)


def _lexicon_matches(
    text: str, lexicon: Sequence[tuple[str, str, bool]]
) -> list[list[Any]]:
    matches: list[list[Any]] = []
    for term, label, _ in lexicon:
        start = text.find(term)
        while start >= 0:
            matches.append([label, start, start + len(term)])
            start = text.find(term, start + len(term))
    return sorted(matches, key=lambda item: (item[1], item[2], item[0]))


def load_fixture(path: Path) -> WorkflowFixture:
    """Load and validate the synthetic workflow fixture."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FixtureError("fixture_unreadable") from exc
    if not isinstance(payload, dict) or payload.get("fixture_version") != 1:
        raise FixtureError("fixture_version_invalid")
    try:
        lexicon = tuple(
            (str(item["term"]), str(item["label"]), bool(item["phi"]))
            for item in payload["lexicon"]
        )
        fixture = WorkflowFixture(
            model_name=str(payload["model_name"]),
            lexicon=lexicon,
            document=str(payload["document"]),
            stream_repeat=int(payload["stream_repeat"]),
            stream_chunk_size=int(payload["stream_chunk_size"]),
            journey_recorded_at=str(payload["journey_recorded_at"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise FixtureError("fixture_invalid") from exc
    if (
        not fixture.lexicon
        or not fixture.phi_terms
        or not fixture.expected_entities
        or fixture.stream_repeat < 2
        or fixture.stream_chunk_size < 1
    ):
        raise FixtureError("fixture_invalid")
    return fixture


def synthetic_loader_class(lexicon: Sequence[tuple[str, str, bool]]) -> type:
    """Return a ``ModelLoader`` stand-in backed by a deterministic lexicon.

    It replaces a cached checkpoint so the gate runs with no model download and
    no ML dependencies; model acquisition is out of scope for the boundary.
    """

    class SyntheticLexiconPipeline:
        openmed_max_length = 512

        def __call__(self, text: Any, **_: Any) -> Any:
            if isinstance(text, (list, tuple)):
                return [self(item) for item in text]
            return [
                {
                    "entity_group": label,
                    "score": 0.99,
                    "start": start,
                    "end": end,
                    "word": str(text)[start:end],
                }
                for label, start, end in _lexicon_matches(str(text), lexicon)
            ]

    class SyntheticLexiconLoader:
        def __init__(self, config: Any = None) -> None:
            self.config = config
            self._pipeline = SyntheticLexiconPipeline()

        def create_pipeline(self, model_name: str, **_: Any) -> Any:
            return self._pipeline

        def get_max_sequence_length(
            self, model_name: str, tokenizer: Any = None
        ) -> int:
            return 512

        def resolve_model_name(self, model_name: str) -> str:
            return model_name

        def loaded_models(self) -> dict[str, Any]:
            return {}

        def unload_model(self, model_name: str) -> dict[str, Any]:
            return {
                "model_name": model_name,
                "models": 0,
                "tokenizers": 0,
                "pipelines": 0,
            }

        def unload_all_models(self) -> dict[str, int]:
            return {"models": 0, "tokenizers": 0, "pipelines": 0}

    return SyntheticLexiconLoader


# ---------------------------------------------------------------------------
# Service process (runs inside the boundary)
# ---------------------------------------------------------------------------


def serve_main(args: argparse.Namespace) -> int:
    """Run the REST (+GraphQL) app and gRPC server on loopback until SIGTERM."""

    fixture = load_fixture(Path(args.fixture))
    from openmed.service import runtime as service_runtime

    service_runtime.ModelLoader = synthetic_loader_class(fixture.lexicon)  # type: ignore[misc]

    import uvicorn

    from openmed.service.app import create_app
    from openmed.service.grpc_server import serve as serve_grpc

    app = create_app()
    grpc_server = serve_grpc(f"127.0.0.1:{args.grpc_port}", max_workers=4)
    # uvicorn re-raises the captured SIGTERM after its graceful shutdown. A
    # no-op handler lets this process stop gRPC too and exit zero, so a zero
    # exit status proves the whole shutdown path completed.
    signal.signal(signal.SIGTERM, lambda *_: None)
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=args.rest_port,
            log_level="warning",
            access_log=False,
            lifespan="on",
        )
    )
    try:
        server.run()
    finally:
        grpc_server.stop(grace=5).wait(10)
    return 0 if server.started else 1


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _service_env(fixture: WorkflowFixture) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in _PROXY_ENV_VARS
        and not key.startswith(("OPENMED_", "HF_", "TRANSFORMERS_"))
    }
    env.update(
        {
            "OPENMED_PROFILE": "prod",
            "OPENMED_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "OPENMED_SERVICE_SERVED_MODELS": fixture.model_name,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    return env


class CaseFailure(Exception):
    """A workflow case failed with a controlled reason code."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class ServiceProcess:
    """One REST + gRPC service process bound to loopback."""

    def __init__(
        self, fixture: WorkflowFixture, fixture_path: Path, *, verbose: bool
    ) -> None:
        self.fixture = fixture
        self.fixture_path = fixture_path
        self.verbose = verbose
        self.process: subprocess.Popen[bytes] | None = None
        self.rest_port = 0
        self.grpc_port = 0

    def start(self, httpx: Any, grpc: Any) -> None:
        """Start the service and wait until REST and gRPC are ready."""

        self.rest_port = _free_port()
        self.grpc_port = _free_port()
        command = [
            sys.executable,
            str(SCRIPT),
            "--serve",
            "--fixture",
            str(self.fixture_path),
            "--rest-port",
            str(self.rest_port),
            "--grpc-port",
            str(self.grpc_port),
        ]
        self.process = subprocess.Popen(
            command,
            env=_service_env(self.fixture),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=None if self.verbose else subprocess.DEVNULL,
        )
        deadline = time.monotonic() + SERVICE_START_TIMEOUT_SECONDS
        with httpx.Client(
            base_url=self.base_url, timeout=5.0, trust_env=False
        ) as client:
            while True:
                if self.process.poll() is not None:
                    raise CaseFailure("server_exit")
                if time.monotonic() > deadline:
                    raise CaseFailure("timeout")
                try:
                    if client.get("/readyz").status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                time.sleep(0.2)
        channel = self.grpc_channel(grpc)
        try:
            grpc.channel_ready_future(channel).result(
                timeout=max(1.0, deadline - time.monotonic())
            )
        except grpc.FutureTimeoutError as exc:
            raise CaseFailure("timeout") from exc
        finally:
            channel.close()

    @property
    def base_url(self) -> str:
        """Return the loopback REST base URL."""

        return f"http://127.0.0.1:{self.rest_port}"

    def grpc_channel(self, grpc: Any) -> Any:
        """Return an insecure loopback channel that never consults proxies."""

        return grpc.insecure_channel(
            f"ipv4:127.0.0.1:{self.grpc_port}",
            options=[("grpc.enable_http_proxy", 0)],
        )

    def stop(self) -> None:
        """Stop gracefully with SIGTERM; raise on timeout or a non-zero exit."""

        process = self.process
        if process is None:
            raise CaseFailure("server_exit")
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=SERVICE_STOP_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired as exc:
                process.kill()
                process.wait()
                raise CaseFailure("timeout") from exc
        self.process = None
        if process.returncode != 0:
            raise CaseFailure("server_exit")

    def kill(self) -> None:
        """Force-stop the process if it is still running."""

        process = self.process
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        self.process = None


def _digest(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _projection(entities: Sequence[Any]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for entity in entities:
        if isinstance(entity, Mapping):
            label, start, end = (
                entity.get("label"),
                entity.get("start"),
                entity.get("end"),
            )
        else:
            label, start, end = entity.label, entity.start, entity.end
        rows.append([str(label), int(start), int(end)])
    return sorted(rows, key=lambda item: (item[1], item[2], item[0]))


def _offsets(rows: Sequence[Sequence[Any]]) -> list[list[int]]:
    return sorted([int(row[1]), int(row[2])] for row in rows)


class Workflow:
    """Drive the synthetic service workflow and collect value-free results."""

    def __init__(
        self,
        fixture: WorkflowFixture,
        fixture_path: Path,
        state_dir: Path,
        *,
        verbose: bool,
    ) -> None:
        import grpc
        import httpx

        from openmed.service.proto.generated import openmed_pb2, openmed_pb2_grpc
        from openmed.structured.store import JobMetadata, SQLiteJourneyStore

        self.grpc = grpc
        self.httpx = httpx
        self.pb2 = openmed_pb2
        self.pb2_grpc = openmed_pb2_grpc
        self.job_metadata = JobMetadata
        self.store_type = SQLiteJourneyStore
        self.fixture = fixture
        self.store_path = state_dir / "journey.sqlite3"
        self.service = ServiceProcess(fixture, fixture_path, verbose=verbose)
        self.results: dict[str, dict[str, Any]] = {}
        self.journey_job_id = (
            "job_" + hashlib.sha256(fixture.document.encode("utf-8")).hexdigest()[:24]
        )
        self.analyze_digest: str | None = None

    # -- bookkeeping -------------------------------------------------------

    def record(
        self, case_id: str, status: str, reason: str, digest: str | None
    ) -> None:
        """Store one case result."""

        self.results[case_id] = {
            "id": case_id,
            "status": status,
            "reason_code": reason,
            "result_digest": digest,
        }

    def run_case(self, case_id: str, operation: Callable[[], str | None]) -> bool:
        """Run one case, mapping every failure to a controlled reason."""

        try:
            digest = operation()
        except CaseFailure as exc:
            self.record(case_id, "failed", exc.reason, None)
            return False
        except self.httpx.TimeoutException:
            self.record(case_id, "failed", "timeout", None)
            return False
        except self.grpc.RpcError as exc:
            code = exc.code() if callable(getattr(exc, "code", None)) else None
            reason = (
                "timeout"
                if code == self.grpc.StatusCode.DEADLINE_EXCEEDED
                else "transport_error"
            )
            self.record(case_id, "failed", reason, None)
            return False
        except Exception:  # never echo exception text
            self.record(case_id, "failed", "transport_error", None)
            return False
        self.record(case_id, "passed", "ok", digest)
        return True

    def skip_remaining(self, case_ids: Sequence[str]) -> None:
        """Mark cases that could not run because a prerequisite failed."""

        for case_id in case_ids:
            if case_id not in self.results:
                self.record(case_id, "not_run", "dependency_failed", None)

    # -- clients -----------------------------------------------------------

    def client(self) -> Any:
        """Return a loopback HTTP client that ignores proxy settings."""

        return self.httpx.Client(
            base_url=self.service.base_url,
            timeout=REQUEST_TIMEOUT_SECONDS,
            trust_env=False,
        )

    def stub(self, channel: Any) -> Any:
        """Return an OpenMed gRPC stub for a loopback channel."""

        return self.pb2_grpc.OpenMedServiceStub(channel)

    def _require_entities(self, rows: list[list[Any]]) -> str:
        if rows != self.fixture.expected_entities:
            raise CaseFailure("unexpected_result")
        return _digest({"entities": rows})

    def _require_offsets(self, rows: list[list[Any]]) -> str:
        if _offsets(rows) != _offsets(self.fixture.expected_entities):
            raise CaseFailure("unexpected_result")
        return _digest({"offsets": _offsets(rows)})

    def _require_redacted(self, text: str, rows: list[list[Any]]) -> str:
        if not text or any(term in text for term in self.fixture.phi_terms):
            raise CaseFailure("unexpected_result")
        if len(rows) < len(self.fixture.phi_terms):
            raise CaseFailure("unexpected_result")
        text_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return _digest({"deidentified_text_sha256": text_digest, "entities": rows})

    @staticmethod
    def _json(response: Any, expected_status: int = 200) -> Any:
        if response.status_code != expected_status:
            raise CaseFailure("unexpected_status")
        try:
            return response.json()
        except ValueError as exc:
            raise CaseFailure("unexpected_result") from exc

    # -- REST --------------------------------------------------------------

    def rest_analyze(self) -> str:
        """POST /analyze and compare entities with the fixture."""

        with self.client() as client:
            payload = self._json(
                client.post(
                    "/analyze",
                    json={
                        "text": self.fixture.document,
                        "model_name": self.fixture.model_name,
                    },
                )
            )
        return self._require_entities(_projection(payload.get("entities", [])))

    def rest_extract(self) -> str:
        """POST /pii/extract and compare entity offsets with the fixture."""

        with self.client() as client:
            payload = self._json(
                client.post(
                    "/pii/extract",
                    json={
                        "text": self.fixture.document,
                        "model_name": self.fixture.model_name,
                    },
                )
            )
        return self._require_offsets(_projection(payload.get("entities", [])))

    def rest_deidentify(self) -> str:
        """POST /pii/deidentify and require every synthetic identifier removed."""

        with self.client() as client:
            payload = self._json(
                client.post(
                    "/pii/deidentify",
                    json={
                        "text": self.fixture.document,
                        "model_name": self.fixture.model_name,
                        "method": "mask",
                    },
                )
            )
        return self._require_redacted(
            str(payload.get("deidentified_text", "")),
            _projection(payload.get("pii_entities", [])),
        )

    def rest_job(self) -> str:
        """Submit an async job and poll until it is done."""

        with self.client() as client:
            created = self._json(
                client.post(
                    "/jobs",
                    json={
                        "documents": [{"text": self.fixture.document}],
                        "model_name": self.fixture.model_name,
                    },
                ),
                expected_status=202,
            )
            job_id = str(created.get("id", ""))
            if not job_id.isalnum():
                raise CaseFailure("unexpected_result")
            deadline = time.monotonic() + JOB_TIMEOUT_SECONDS
            while True:
                record = self._json(client.get(f"/jobs/{job_id}"))
                if record.get("status") in {"done", "failed"}:
                    break
                if time.monotonic() > deadline:
                    raise CaseFailure("timeout")
                time.sleep(0.1)
        if record.get("status") != "done" or record.get("processed_count") != 1:
            raise CaseFailure("unexpected_result")
        histogram = record.get("label_histogram")
        if not isinstance(histogram, dict) or not histogram:
            raise CaseFailure("unexpected_result")
        return _digest({"label_histogram": histogram, "processed_count": 1})

    def rest_webhook_egress_denied(self) -> None:
        """Require the in-process egress policy to refuse an external webhook."""

        with self.client() as client:
            response = client.post(
                "/jobs",
                json={
                    "documents": [{"text": self.fixture.document}],
                    "model_name": self.fixture.model_name,
                    "webhook": {
                        "url": f"https://{PROBE_IPV4}/openmed-no-egress-probe",
                        "secret": "synthetic-no-egress-webhook-secret",
                    },
                },
            )
        payload = self._json(response, expected_status=403)
        error = payload.get("error") if isinstance(payload, dict) else None
        if not isinstance(error, dict) or error.get("code") != "egress_denied":
            raise CaseFailure("unexpected_result")
        return None

    def rest_stream_cancel(self) -> None:
        """Abandon a de-identification stream mid-flight; the service must recover."""

        text = " ".join([self.fixture.document] * self.fixture.stream_repeat)
        with self.client() as client:
            with client.stream(
                "POST",
                "/pii/deidentify/stream",
                json={
                    "text": text,
                    "model_name": self.fixture.model_name,
                    "chunk_size": self.fixture.stream_chunk_size,
                },
            ) as response:
                if response.status_code != 200:
                    raise CaseFailure("unexpected_status")
                first = next(response.iter_lines(), "")
                try:
                    event = json.loads(first)
                except ValueError as exc:
                    raise CaseFailure("unexpected_result") from exc
                if not isinstance(event, dict) or event.get("type") == "final":
                    raise CaseFailure("not_cancelled")
            # Leaving the block closes the connection before the stream ends.
            if client.get("/readyz").status_code != 200:
                raise CaseFailure("unexpected_status")
        if self.rest_analyze() != self.analyze_digest:
            raise CaseFailure("unexpected_result")
        return None

    # -- GraphQL -----------------------------------------------------------

    def _graphql(self, query: str, variables: Mapping[str, Any]) -> Any:
        with self.client() as client:
            payload = self._json(
                client.post("/graphql", json={"query": query, "variables": variables})
            )
        if not isinstance(payload, dict) or payload.get("errors"):
            raise CaseFailure("unexpected_result")
        return payload.get("data") or {}

    def graphql_analyze(self) -> str:
        """Run the GraphQL ``analyze`` query."""

        data = self._graphql(
            "query NoEgressAnalyze($input: AnalyzeInput!) {"
            " analyze(input: $input) { entities { label start end } } }",
            {
                "input": {
                    "text": self.fixture.document,
                    "modelName": self.fixture.model_name,
                }
            },
        )
        entities = (data.get("analyze") or {}).get("entities", [])
        return self._require_entities(_projection(entities))

    def graphql_deidentify(self) -> str:
        """Run the GraphQL ``deidentify`` query."""

        data = self._graphql(
            "query NoEgressDeidentify($input: DeidentifyInput!) {"
            " deidentify(input: $input) {"
            " deidentifiedText entities { label start end } } }",
            {
                "input": {
                    "text": self.fixture.document,
                    "modelName": self.fixture.model_name,
                }
            },
        )
        result = data.get("deidentify") or {}
        return self._require_redacted(
            str(result.get("deidentifiedText", "")),
            _projection(result.get("entities", [])),
        )

    # -- gRPC --------------------------------------------------------------

    def grpc_analyze(self) -> str:
        """Call the unary ``Analyze`` RPC."""

        channel = self.service.grpc_channel(self.grpc)
        try:
            response = self.stub(channel).Analyze(
                self.pb2.AnalyzeRequest(
                    text=self.fixture.document, model_name=self.fixture.model_name
                ),
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        finally:
            channel.close()
        return self._require_entities(_projection(list(response.entities)))

    def grpc_extract(self) -> str:
        """Call the unary ``Extract`` RPC."""

        channel = self.service.grpc_channel(self.grpc)
        try:
            response = self.stub(channel).Extract(
                self.pb2.ExtractRequest(
                    text=self.fixture.document, model_name=self.fixture.model_name
                ),
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        finally:
            channel.close()
        return self._require_offsets(_projection(list(response.entities)))

    def grpc_deidentify(self) -> str:
        """Call the unary ``Deidentify`` RPC."""

        channel = self.service.grpc_channel(self.grpc)
        try:
            response = self.stub(channel).Deidentify(
                self.pb2.DeidentifyRequest(
                    text=self.fixture.document,
                    model_name=self.fixture.model_name,
                    method="mask",
                ),
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        finally:
            channel.close()
        return self._require_redacted(
            response.deidentified_text, _projection(list(response.pii_entities))
        )

    def grpc_stream_cancel(self) -> None:
        """Cancel a server stream after its first message; the server must recover."""

        text = " ".join([self.fixture.document] * self.fixture.stream_repeat)
        channel = self.service.grpc_channel(self.grpc)
        try:
            call = self.stub(channel).StreamDeidentify(
                self.pb2.DeidentifyStreamRequest(
                    request=self.pb2.DeidentifyRequest(
                        text=text,
                        model_name=self.fixture.model_name,
                        method="mask",
                    ),
                    chunk_size=self.fixture.stream_chunk_size,
                    max_buffer=self.fixture.stream_chunk_size,
                ),
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            first = next(call)
            if first.final:
                raise CaseFailure("not_cancelled")
            call.cancel()
            try:
                for _ in call:
                    pass
            except self.grpc.RpcError as exc:
                if exc.code() != self.grpc.StatusCode.CANCELLED:
                    raise CaseFailure("unexpected_status") from exc
            else:
                raise CaseFailure("not_cancelled")
        finally:
            channel.close()
        if self.grpc_analyze() != self.analyze_digest:
            raise CaseFailure("unexpected_result")
        return None

    # -- Journey -----------------------------------------------------------

    def journey_persist(self) -> str:
        """Append PHI-free job metadata to a local Journey store."""

        store = self.store_type(self.store_path)
        try:
            result = store.put_job(
                self.job_metadata(
                    job_id=self.journey_job_id,
                    state="completed",
                    recorded_at=self.fixture.journey_recorded_at,
                    metadata={
                        "document_count": 1,
                        "entity_count": len(self.fixture.expected_entities),
                    },
                ),
                committed_at=self.fixture.journey_recorded_at,
            )
            if not result.ok or not store.integrity_check().ok:
                raise CaseFailure("persistence_mismatch")
            job = store.get_job(self.journey_job_id)
            if not job.ok or job.value is None:
                raise CaseFailure("persistence_mismatch")
            return _digest({"job": job.value.to_dict()})
        finally:
            store.close()

    def journey_reopen(self) -> str:
        """Reopen the store after the restart and verify the committed record."""

        opened = self.store_type.open(self.store_path)
        if not opened.ok or opened.value is None:
            raise CaseFailure("persistence_mismatch")
        store = opened.value
        try:
            if not store.integrity_check().ok:
                raise CaseFailure("persistence_mismatch")
            versions = store.list_job_versions(self.journey_job_id)
            if not versions.ok or versions.value is None or len(versions.value) != 1:
                raise CaseFailure("persistence_mismatch")
            if versions.value[0].state != "completed":
                raise CaseFailure("persistence_mismatch")
            appended = store.put_job(
                self.job_metadata(
                    job_id=self.journey_job_id,
                    state="verified_after_restart",
                    recorded_at=self.fixture.journey_recorded_at,
                    metadata={"document_count": 1},
                ),
                committed_at=self.fixture.journey_recorded_at,
            )
            if not appended.ok:
                raise CaseFailure("persistence_mismatch")
            final = store.list_job_versions(self.journey_job_id)
            if not final.ok or final.value is None or len(final.value) != 2:
                raise CaseFailure("persistence_mismatch")
            return _digest({"versions": [item.to_dict() for item in final.value]})
        finally:
            store.close()

    # -- lifecycle ---------------------------------------------------------

    def service_start(self) -> None:
        """Start the service on loopback."""

        self.service.start(self.httpx, self.grpc)
        return None

    def service_restart(self) -> str:
        """Restart gracefully; results must be identical across the restart."""

        self.service.stop()
        self.service.start(self.httpx, self.grpc)
        digest = self.rest_analyze()
        if digest != self.analyze_digest:
            raise CaseFailure("unexpected_result")
        return digest

    def service_shutdown(self) -> None:
        """Stop gracefully; the process must exit zero."""

        self.service.stop()
        return None

    def run(self) -> list[dict[str, Any]]:
        """Run every case in order and return the value-free results."""

        from openmed.core.attestation import NO_EGRESS_WORKFLOW_CASES

        order = list(NO_EGRESS_WORKFLOW_CASES)
        try:
            if not self.run_case("service_start", self.service_start):
                self.skip_remaining(order)
                return [self.results[case] for case in order]
            if self.run_case("rest_analyze", self.rest_analyze):
                self.analyze_digest = self.results["rest_analyze"]["result_digest"]
            for case_id, operation in (
                ("rest_extract", self.rest_extract),
                ("rest_deidentify", self.rest_deidentify),
                ("rest_job", self.rest_job),
                ("rest_webhook_egress_denied", self.rest_webhook_egress_denied),
                ("rest_stream_cancel", self.rest_stream_cancel),
                ("graphql_analyze", self.graphql_analyze),
                ("graphql_deidentify", self.graphql_deidentify),
                ("grpc_analyze", self.grpc_analyze),
                ("grpc_extract", self.grpc_extract),
                ("grpc_deidentify", self.grpc_deidentify),
                ("grpc_stream_cancel", self.grpc_stream_cancel),
            ):
                self.run_case(case_id, operation)
            persisted = self.run_case("journey_persist", self.journey_persist)
            restarted = self.run_case("service_restart", self.service_restart)
            if persisted:
                # The store was closed and reopened across the service restart.
                self.run_case("journey_reopen", self.journey_reopen)
            if restarted:
                self.run_case("service_shutdown", self.service_shutdown)
            self.skip_remaining(order)
        finally:
            self.service.kill()
        return [self.results[case] for case in order]


def run_workflow(
    fixture: WorkflowFixture, fixture_path: Path, *, verbose: bool
) -> list[dict[str, Any]]:
    """Run the workflow in a private state directory that is always removed."""

    state_dir = Path(tempfile.mkdtemp(prefix="openmed-no-egress-"))
    try:
        try:
            workflow = Workflow(fixture, fixture_path, state_dir, verbose=verbose)
        except ImportError:
            from openmed.core.attestation import NO_EGRESS_WORKFLOW_CASES

            return [
                {
                    "id": case_id,
                    "status": "failed",
                    "reason_code": "dependency_unavailable",
                    "result_digest": None,
                }
                for case_id in NO_EGRESS_WORKFLOW_CASES
            ]
        return workflow.run()
    finally:
        shutil.rmtree(state_dir, ignore_errors=True)


def inside_main(args: argparse.Namespace) -> int:
    """Run inside the boundary and print one value-free result line."""

    fixture_path = Path(args.fixture)
    fixture = load_fixture(fixture_path)
    if args.boundary_kind == "netns":
        bring_loopback_up()
    interfaces, routes = inspect_boundary()
    with contextlib.closing(LoopbackResponders()) as responders:
        positives = run_probe_set(positive_probes(responders), classify_positive)
    # Negative controls run before openmed is imported in this process.
    negatives = run_probe_set(negative_probes(), classify_negative)
    workflow = run_workflow(fixture, fixture_path, verbose=args.verbose)
    result = {
        "kernel": "linux" if sys.platform.startswith("linux") else "unknown",
        "interfaces": interfaces,
        "routes": routes,
        "negative_controls": negatives,
        "positive_controls": positives,
        "workflow": workflow,
    }
    sys.stdout.write(RESULT_MARKER + json.dumps(result, sort_keys=True) + "\n")
    sys.stdout.flush()
    return 0


# ---------------------------------------------------------------------------
# Host side: boundary selection, runner, attestation
# ---------------------------------------------------------------------------

CommandRunner = Callable[[Sequence[str], float], int]


def run_quiet(command: Sequence[str], timeout: float) -> int:
    """Run a probe command with no output; return its exit code or ``-1``."""

    try:
        return subprocess.run(
            list(command),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
        ).returncode
    except (OSError, subprocess.SubprocessError):
        return -1


@dataclass(frozen=True)
class BoundaryPlan:
    """The selected boundary, or the controlled reason none is available."""

    kind: str
    skip_reason: str | None = None
    executable: str | None = None
    image: str | None = None


def host_kind() -> str:
    """Return the controlled host kind for this machine."""

    system = platform.system().lower()
    return system if system in {"linux", "darwin", "windows"} else "other"


def netns_executable(runner: CommandRunner = run_quiet) -> str | None:
    """Return ``unshare`` when it can create a user + network namespace."""

    unshare = shutil.which("unshare")
    if unshare is None:
        return None
    command = [
        unshare,
        "--user",
        "--map-root-user",
        "--net",
        "--",
        sys.executable,
        "-c",
        "pass",
    ]
    return unshare if runner(command, 30.0) == 0 else None


def container_executable(
    image: str | None, runner: CommandRunner = run_quiet
) -> str | None:
    """Return ``docker`` when the daemon is reachable and ``image`` is local."""

    docker = shutil.which("docker")
    if docker is None or not image:
        return None
    if runner([docker, "info"], 30.0) != 0:
        return None
    if runner([docker, "image", "inspect", image], 30.0) != 0:
        return None
    return docker


def select_boundary(
    requested: str,
    host: str,
    image: str | None,
    *,
    runner: CommandRunner = run_quiet,
) -> BoundaryPlan:
    """Pick an OS boundary or return a controlled skip.

    ``netns`` needs a Linux host that permits unprivileged user namespaces.
    ``container`` needs a reachable Docker daemon and a local image with the
    service dependencies (the gate never pulls). ``auto`` tries ``netns`` on
    Linux, then ``container`` when an image is configured.
    """

    if requested in {"netns", "auto"} and host == "linux":
        unshare = netns_executable(runner)
        if unshare is not None:
            return BoundaryPlan("netns", executable=unshare)
    if requested == "netns":
        reason = "unsupported_host" if host != "linux" else "boundary_unavailable"
        return BoundaryPlan("none", skip_reason=reason)
    if requested == "container" or image:
        docker = container_executable(image, runner)
        if docker is not None:
            return BoundaryPlan("container", executable=docker, image=image)
        if requested == "container":
            return BoundaryPlan("none", skip_reason="boundary_unavailable")
    reason = "boundary_unavailable" if host == "linux" else "unsupported_host"
    return BoundaryPlan("none", skip_reason=reason)


def inner_command(
    plan: BoundaryPlan, fixture: Path, *, verbose: bool, container_name: str
) -> list[str]:
    """Return the command that runs the inner workload inside the boundary."""

    extra = ["--verbose"] if verbose else []
    if plan.kind == "netns" and plan.executable:
        return [
            plan.executable,
            "--user",
            "--map-root-user",
            "--net",
            "--",
            sys.executable,
            str(SCRIPT),
            "--inside",
            "--boundary-kind",
            "netns",
            "--fixture",
            str(fixture),
            *extra,
        ]
    if plan.kind == "container" and plan.executable and plan.image:
        return [
            plan.executable,
            "run",
            "--rm",
            "--name",
            container_name,
            "--network",
            "none",
            "--pull",
            "never",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=256m",
            "--user",
            "65534:65534",
            "--mount",
            f"type=bind,src={ROOT},dst={CONTAINER_WORKSPACE},readonly",
            "--mount",
            f"type=bind,src={fixture},dst={CONTAINER_FIXTURE},readonly",
            "--workdir",
            "/tmp",
            "--env",
            "HOME=/tmp",
            "--env",
            "TMPDIR=/tmp",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--env",
            "PYTHONUNBUFFERED=1",
            plan.image,
            "python",
            f"{CONTAINER_WORKSPACE}/scripts/security/{SCRIPT.name}",
            "--inside",
            "--boundary-kind",
            "container",
            "--fixture",
            CONTAINER_FIXTURE,
            *extra,
        ]
    raise ValueError("boundary_plan_invalid")


@dataclass(frozen=True)
class InnerOutcome:
    """The in-boundary runner's parsed result or a controlled failure."""

    result: Mapping[str, Any] | None
    failure: str | None


def parse_inner_output(stdout: str, returncode: int) -> InnerOutcome:
    """Extract the single marked result line printed by the inner runner."""

    lines = [line for line in stdout.splitlines() if line.startswith(RESULT_MARKER)]
    if not lines:
        return InnerOutcome(None, "runner_failed")
    try:
        payload = json.loads(lines[-1][len(RESULT_MARKER) :])
    except json.JSONDecodeError:
        return InnerOutcome(None, "runner_result_invalid")
    required = {
        "kernel",
        "interfaces",
        "routes",
        "negative_controls",
        "positive_controls",
        "workflow",
    }
    if not isinstance(payload, dict) or set(payload) != required or returncode != 0:
        return InnerOutcome(None, "runner_result_invalid")
    return InnerOutcome(payload, None)


def run_inner(
    plan: BoundaryPlan,
    fixture: Path,
    *,
    verbose: bool,
    timeout: float,
) -> InnerOutcome:
    """Run the inner workload inside the boundary and parse its result."""

    container_name = f"openmed-no-egress-{uuid.uuid4().hex[:12]}"
    command = inner_command(
        plan, fixture, verbose=verbose, container_name=container_name
    )
    env = {
        key: value for key, value in os.environ.items() if key not in _PROXY_ENV_VARS
    }
    try:
        completed = subprocess.run(
            command,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=None if verbose else subprocess.DEVNULL,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        if plan.kind == "container" and plan.executable:
            run_quiet([plan.executable, "rm", "--force", container_name], 60.0)
        return InnerOutcome(None, "runner_timeout")
    except OSError:
        return InnerOutcome(None, "runner_failed")
    return parse_inner_output(completed.stdout or "", completed.returncode)


def file_digest(path: Path) -> str:
    """Return the ``sha256:`` digest of a file's bytes."""

    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_commit(explicit: str | None) -> str | None:
    """Return the commit from ``--commit``, ``GITHUB_SHA``, or ``git``."""

    for candidate in (explicit, os.environ.get("GITHUB_SHA")):
        if candidate:
            return candidate.strip().lower()
    git = shutil.which("git")
    if git is None:
        return None
    try:
        completed = subprocess.run(
            [git, "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip().lower() if completed.returncode == 0 else None


def build_attestation(
    *,
    plan: BoundaryPlan,
    requested: str,
    host: str,
    fixtures: Mapping[str, str],
    outcome: InnerOutcome | None,
    commit: str | None,
    generated_at: str | None,
) -> Any:
    """Turn a plan and inner outcome into a validated attestation."""

    from openmed.core.attestation import (
        NoEgressAttestationError,
        build_no_egress_attestation,
    )

    common: dict[str, Any] = {
        "host_kind": host,
        "requested_boundary": requested,
        "fixtures": fixtures,
        "commit": commit,
        "generated_at": generated_at,
    }
    if plan.skip_reason is not None:
        return build_no_egress_attestation(
            boundary_kind="none", skip_reason=plan.skip_reason, **common
        )
    if outcome is None or outcome.result is None:
        return build_no_egress_attestation(
            boundary_kind=plan.kind,
            runner_failure=(outcome.failure if outcome else None) or "runner_failed",
            **common,
        )
    result = outcome.result
    try:
        return build_no_egress_attestation(
            boundary_kind=plan.kind,
            kernel=result["kernel"],
            interfaces=result["interfaces"],
            routes=result["routes"],
            negative_controls=result["negative_controls"],
            positive_controls=result["positive_controls"],
            workflow=result["workflow"],
            **common,
        )
    except (NoEgressAttestationError, KeyError, TypeError):
        return build_no_egress_attestation(
            boundary_kind=plan.kind,
            runner_failure="runner_result_invalid",
            **common,
        )


def write_output(attestation: Any, output: Path | None, markdown: Path | None) -> None:
    """Write the JSON attestation (and optional Markdown) or print to stdout."""

    text = attestation.to_json(indent=2) + "\n"
    if output is None:
        sys.stdout.write(text)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    if markdown is not None:
        markdown.parent.mkdir(parents=True, exist_ok=True)
        markdown.write_text(attestation.to_markdown(), encoding="utf-8")


def gate_main(args: argparse.Namespace) -> int:
    """Run the gate on the host and return its exit code."""

    fixture_path = Path(args.fixture).expanduser().resolve()
    try:
        load_fixture(fixture_path)
    except FixtureError as exc:
        print(f"no-egress gate: {exc}", file=sys.stderr)
        return EXIT_USAGE
    fixtures = {
        "service_workflow_fixture": file_digest(fixture_path),
        "no_egress_gate_runner": file_digest(SCRIPT),
    }
    host = host_kind()
    image = args.container_image or os.environ.get(IMAGE_ENV_VAR) or None
    plan = select_boundary(args.boundary, host, image)
    outcome = None
    if plan.skip_reason is None:
        outcome = run_inner(
            plan, fixture_path, verbose=args.verbose, timeout=args.timeout
        )
    attestation = build_attestation(
        plan=plan,
        requested=args.boundary,
        host=host,
        fixtures=fixtures,
        outcome=outcome,
        commit=resolve_commit(args.commit),
        generated_at=args.generated_at,
    )
    write_output(attestation, args.output, args.markdown)
    payload = attestation.to_dict()
    counts = payload["counts"]
    print(
        "no-egress gate: "
        f"status={payload['status']} reason={payload['reason_code']} "
        f"boundary={payload['boundary']['kind']} "
        f"negative_blocked={counts['negative_controls']['blocked']}/"
        f"{counts['negative_controls']['total']} "
        f"workflow_passed={counts['workflow']['passed']}/"
        f"{counts['workflow']['total']}",
        file=sys.stderr,
    )
    return int(attestation.exit_code)


def verify_main(args: argparse.Namespace) -> int:
    """Validate a saved attestation; exit with the code its status implies."""

    from openmed.core.attestation import (
        NoEgressAttestationError,
        validate_no_egress_attestation,
    )

    try:
        payload = json.loads(Path(args.verify).read_text(encoding="utf-8"))
        attestation = validate_no_egress_attestation(payload)
    except (OSError, json.JSONDecodeError):
        print("no-egress verify: attestation_unreadable", file=sys.stderr)
        return EXIT_FAILED
    except NoEgressAttestationError as exc:
        print(f"no-egress verify: {exc.code}", file=sys.stderr)
        return EXIT_FAILED
    print(
        f"no-egress verify: valid status={attestation.status} "
        f"reason={attestation.reason_code}",
        file=sys.stderr,
    )
    return int(attestation.exit_code)


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Run a synthetic OpenMed service workflow inside an OS-enforced "
            "no-egress boundary and write a value-free attestation."
        )
    )
    parser.add_argument(
        "--boundary",
        choices=("auto", "netns", "container"),
        default="auto",
        help="Boundary mechanism (default: auto).",
    )
    parser.add_argument(
        "--container-image",
        default=None,
        help=(
            "Local image with Python 3.11+ and the service extra for container "
            f"mode; defaults to ${IMAGE_ENV_VAR}. The gate never pulls."
        ),
    )
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)
    parser.add_argument("--commit", default=None)
    parser.add_argument(
        "--generated-at",
        default=None,
        help="Declared ISO-8601 timestamp; the only volatile field.",
    )
    parser.add_argument("--timeout", type=float, default=DEFAULT_RUNNER_TIMEOUT_SECONDS)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--verify", type=Path, default=None)
    parser.add_argument("--inside", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--serve", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--boundary-kind",
        choices=("netns", "container"),
        default="netns",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--rest-port", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--grpc-port", type=int, default=0, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""

    args = build_parser().parse_args(argv)
    if args.serve:
        return serve_main(args)
    if args.inside:
        return inside_main(args)
    if args.verify is not None:
        return verify_main(args)
    return gate_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
