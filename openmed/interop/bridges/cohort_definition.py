"""Optional JSON-over-stdin bridge for open cohort-definition services.

The bridge is deliberately transport-neutral and out of process.  OpenMed
does not bundle a service, executable, vocabulary, or network client.  A user
may supply an executable adapter, or inject a callable in tests/applications,
that implements the documented versioned protocol.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from openmed.clinical.journey_contracts import canonical_digest
from openmed.structured.cohort.exchange import (
    CohortConversionLoss,
    CohortDefinitionExchange,
)
from openmed.structured.store import StoreResult, StoreState

COHORT_SERVICE_PROTOCOL_VERSION: Final = "1.0.0"
COHORT_SERVICE_COMPATIBILITY_POLICY: Final = "same_major"
DEFAULT_COHORT_SERVICE_TIMEOUT_SECONDS: Final = 120.0
MAX_COHORT_SERVICE_RESPONSE_BYTES: Final = 8_000_000

_ALLOWED_STATES: Final = frozenset(StoreState)


class CohortServiceBridgeError(RuntimeError):
    """Base error for adapter execution and protocol failures."""


class CohortServiceUnavailableError(CohortServiceBridgeError):
    """Raised when no configured adapter can be resolved."""


class CohortServiceProtocolError(CohortServiceBridgeError):
    """Raised when an adapter response violates the bridge contract."""


Runner = Callable[[Mapping[str, Any]], Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class CohortServiceConversion:
    """External target plus digest-bound, explicit conversion losses."""

    direction: str
    source_digest: str
    target: Mapping[str, Any] | None
    losses: tuple[CohortConversionLoss, ...] = ()
    adapter_name: str = "open-cohort-service"
    adapter_version: str = "unknown"
    protocol_version: str = COHORT_SERVICE_PROTOCOL_VERSION
    compatibility_policy: str = COHORT_SERVICE_COMPATIBILITY_POLICY

    def __post_init__(self) -> None:
        if self.direction not in {"export", "import"}:
            raise CohortServiceProtocolError("conversion direction is unsupported")
        if self.protocol_version != COHORT_SERVICE_PROTOCOL_VERSION:
            raise CohortServiceProtocolError("protocol version is unsupported")
        if self.compatibility_policy != COHORT_SERVICE_COMPATIBILITY_POLICY:
            raise CohortServiceProtocolError("compatibility policy is unsupported")
        if not isinstance(self.source_digest, str) or not self.source_digest.startswith(
            "sha256:"
        ):
            raise CohortServiceProtocolError("source digest is invalid")
        if self.target is not None and not isinstance(self.target, Mapping):
            raise CohortServiceProtocolError("target must be an object or null")
        if not isinstance(self.adapter_name, str) or not self.adapter_name:
            raise CohortServiceProtocolError("adapter name is invalid")
        if not isinstance(self.adapter_version, str) or not self.adapter_version:
            raise CohortServiceProtocolError("adapter version is invalid")
        object.__setattr__(
            self,
            "target",
            None if self.target is None else dict(self.target),
        )
        object.__setattr__(
            self,
            "losses",
            tuple(sorted(self.losses, key=lambda item: (item.path, item.reason_code))),
        )

    @property
    def lossless(self) -> bool:
        """Return whether the adapter declared no conversion loss."""

        return not self.losses

    def to_dict(self) -> dict[str, Any]:
        """Return the complete adapter receipt."""

        return {
            "adapter_name": self.adapter_name,
            "adapter_version": self.adapter_version,
            "compatibility_policy": self.compatibility_policy,
            "direction": self.direction,
            "losses": [item.to_dict() for item in self.losses],
            "lossless": self.lossless,
            "protocol_version": self.protocol_version,
            "source_digest": self.source_digest,
            "target": None if self.target is None else dict(self.target),
        }


@dataclass(frozen=True, slots=True)
class CohortDefinitionServiceBridge:
    """Invoke a caller-supplied cohort-definition adapter safely.

    ``command`` is executed without a shell.  The request is written to stdin,
    stderr is discarded to prevent a third-party adapter from echoing cohort
    content, and only a bounded JSON response is accepted.  ``runner`` is an
    equivalent dependency-injection surface for an already isolated service.
    Exactly one transport may be configured.
    """

    command: tuple[str, ...] | None = None
    runner: Runner | None = None
    timeout_seconds: float = DEFAULT_COHORT_SERVICE_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        if self.command is not None and self.runner is not None:
            raise ValueError("configure command or runner, not both")
        if self.command is not None and (
            not self.command
            or any(not isinstance(part, str) or not part for part in self.command)
        ):
            raise ValueError("command must contain non-empty strings")
        if self.runner is not None and not callable(self.runner):
            raise TypeError("runner must be callable")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or self.timeout_seconds <= 0
            or not math.isfinite(self.timeout_seconds)
        ):
            raise ValueError("timeout_seconds must be positive")

    @property
    def available(self) -> bool:
        """Return whether the configured transport is currently usable."""

        if self.runner is not None:
            return True
        if self.command is None:
            return False
        executable = self.command[0]
        path = Path(executable)
        if path.is_absolute() or path.parent != Path("."):
            return path.is_file() and os.access(path, os.X_OK)
        return shutil.which(executable) is not None

    def export(
        self,
        envelope: CohortDefinitionExchange,
        *,
        target_format: str,
        strict: bool = False,
    ) -> StoreResult[CohortServiceConversion]:
        """Convert an OpenMed envelope to a named external JSON format."""

        if not isinstance(envelope, CohortDefinitionExchange):
            return StoreResult.outcome(
                StoreState.FAILURE, "cohort_bridge_input_invalid"
            )
        if not _format_name(target_format):
            return StoreResult.outcome(
                StoreState.UNSUPPORTED, "cohort_bridge_format_unsupported"
            )
        request = _request(
            direction="export",
            source_format="openmed.cohort.exchange",
            target_format=target_format,
            payload=envelope.to_dict(),
        )
        return self._convert(request, strict=strict)

    def import_definition(
        self,
        payload: Mapping[str, Any],
        *,
        source_format: str,
        strict: bool = False,
    ) -> StoreResult[CohortServiceConversion]:
        """Convert an external JSON definition to the OpenMed envelope shape."""

        if not isinstance(payload, Mapping):
            return StoreResult.outcome(
                StoreState.FAILURE, "cohort_bridge_input_invalid"
            )
        if not _format_name(source_format):
            return StoreResult.outcome(
                StoreState.UNSUPPORTED, "cohort_bridge_format_unsupported"
            )
        request = _request(
            direction="import",
            source_format=source_format,
            target_format="openmed.cohort.exchange",
            payload=payload,
        )
        return self._convert(request, strict=strict)

    def _convert(
        self, request: Mapping[str, Any], *, strict: bool
    ) -> StoreResult[CohortServiceConversion]:
        if not self.available:
            return StoreResult.outcome(StoreState.UNKNOWN, "cohort_bridge_unavailable")
        try:
            response = self._invoke(request)
            state, conversion = _parse_response(response, request=request)
            if conversion.losses and state is StoreState.SUCCESS:
                state = StoreState.PARTIAL
            if strict and conversion.losses:
                state = StoreState.UNSUPPORTED
            if state is StoreState.SUCCESS:
                return StoreResult.success(conversion)
            return StoreResult.outcome(
                state,
                _result_code(state),
                value=conversion,
            )
        except CohortServiceUnavailableError:
            return StoreResult.outcome(StoreState.UNKNOWN, "cohort_bridge_unavailable")
        except CohortServiceProtocolError:
            return StoreResult.outcome(
                StoreState.FAILURE, "cohort_bridge_protocol_invalid"
            )
        except CohortServiceBridgeError:
            return StoreResult.outcome(
                StoreState.FAILURE, "cohort_bridge_execution_failed"
            )
        except (OSError, subprocess.SubprocessError, TypeError, ValueError):
            return StoreResult.outcome(
                StoreState.FAILURE, "cohort_bridge_execution_failed"
            )

    def _invoke(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        if self.runner is not None:
            try:
                response = self.runner(dict(request))
            except Exception:
                raise CohortServiceBridgeError("cohort runner failed") from None
            if not isinstance(response, Mapping):
                raise CohortServiceProtocolError("runner response must be an object")
            return response
        if self.command is None or not self.available:
            raise CohortServiceUnavailableError("cohort adapter is unavailable")
        encoded = json.dumps(
            request,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        raw_output = _run_bounded_adapter(
            self.command,
            encoded,
            timeout=float(self.timeout_seconds),
            environment=_safe_environment(),
        )
        try:
            response = json.loads(raw_output.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise CohortServiceProtocolError(
                "cohort adapter returned invalid JSON"
            ) from None
        if not isinstance(response, Mapping):
            raise CohortServiceProtocolError("adapter response must be an object")
        return response


def _run_bounded_adapter(
    argv: tuple[str, ...],
    payload: bytes,
    *,
    timeout: float,
    environment: Mapping[str, str],
) -> bytes:
    """Bound captured output while feeding input and reap the owned process."""

    process = subprocess.Popen(  # noqa: S603 - explicit caller-selected argv
        argv,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env=environment,
    )
    assert process.stdin is not None and process.stdout is not None
    timed_out = threading.Event()

    def expire() -> None:
        timed_out.set()
        if process.poll() is None:
            process.kill()

    def feed() -> None:
        try:
            process.stdin.write(payload)
            process.stdin.flush()
        except (BrokenPipeError, OSError):
            pass
        finally:
            try:
                process.stdin.close()
            except OSError:
                pass

    writer = threading.Thread(target=feed, name="openmed-cohort-input")
    watchdog = threading.Timer(timeout, expire)
    output = bytearray()
    try:
        writer.start()
        watchdog.start()
        while chunk := process.stdout.read1(
            min(65536, MAX_COHORT_SERVICE_RESPONSE_BYTES + 1 - len(output))
        ):
            output.extend(chunk)
            if len(output) > MAX_COHORT_SERVICE_RESPONSE_BYTES:
                raise CohortServiceProtocolError(
                    "cohort adapter output size is invalid"
                )
        process.wait()
        if timed_out.is_set():
            raise subprocess.TimeoutExpired(argv, timeout)
        if process.returncode != 0:
            raise CohortServiceBridgeError("cohort adapter failed")
        return bytes(output)
    finally:
        watchdog.cancel()
        if process.poll() is None:
            process.kill()
        process.wait()
        if writer.ident is not None:
            writer.join()
        else:
            process.stdin.close()
        if watchdog.ident is not None:
            watchdog.join()
        process.stdout.close()


def _request(
    *,
    direction: str,
    source_format: str,
    target_format: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "compatibility_policy": COHORT_SERVICE_COMPATIBILITY_POLICY,
        "direction": direction,
        "payload": dict(payload),
        "protocol_version": COHORT_SERVICE_PROTOCOL_VERSION,
        "source_digest": canonical_digest(payload),
        "source_format": source_format,
        "target_format": target_format,
    }


def _parse_response(
    value: Mapping[str, Any], *, request: Mapping[str, Any]
) -> tuple[StoreState, CohortServiceConversion]:
    required = {
        "adapter_name",
        "adapter_version",
        "compatibility_policy",
        "direction",
        "losses",
        "protocol_version",
        "source_digest",
        "state",
        "target",
    }
    if set(value) != required:
        raise CohortServiceProtocolError("adapter response fields are incompatible")
    if value["protocol_version"] != COHORT_SERVICE_PROTOCOL_VERSION:
        raise CohortServiceProtocolError("adapter protocol version is unsupported")
    if value["compatibility_policy"] != COHORT_SERVICE_COMPATIBILITY_POLICY:
        raise CohortServiceProtocolError("adapter compatibility policy is unsupported")
    if value["direction"] != request["direction"]:
        raise CohortServiceProtocolError("adapter response direction differs")
    if value["source_digest"] != request["source_digest"]:
        raise CohortServiceProtocolError("adapter source digest differs")
    try:
        state = StoreState(value["state"])
    except (TypeError, ValueError):
        raise CohortServiceProtocolError("adapter state is unsupported") from None
    if state not in _ALLOWED_STATES:
        raise CohortServiceProtocolError("adapter state is unsupported")
    losses_value = value["losses"]
    if isinstance(losses_value, (str, bytes)) or not isinstance(losses_value, Sequence):
        raise CohortServiceProtocolError("adapter losses must be an array")
    losses = tuple(
        CohortConversionLoss.from_dict(item)
        for item in losses_value
        if isinstance(item, Mapping)
    )
    if len(losses) != len(losses_value):
        raise CohortServiceProtocolError("adapter loss is invalid")
    target = value["target"]
    if target is not None and not isinstance(target, Mapping):
        raise CohortServiceProtocolError("adapter target must be an object or null")
    if state in {StoreState.SUCCESS, StoreState.PARTIAL} and target is None:
        raise CohortServiceProtocolError("successful adapter response needs a target")
    if state is StoreState.SUCCESS and losses:
        state = StoreState.PARTIAL
    return state, CohortServiceConversion(
        direction=str(value["direction"]),
        source_digest=str(value["source_digest"]),
        target=target,
        losses=losses,
        adapter_name=str(value["adapter_name"]),
        adapter_version=str(value["adapter_version"]),
        protocol_version=str(value["protocol_version"]),
        compatibility_policy=str(value["compatibility_policy"]),
    )


def _format_name(value: str) -> bool:
    if not isinstance(value, str) or not value or len(value) > 128:
        return False
    return all(character.isalnum() or character in "._-/" for character in value)


def _result_code(state: StoreState) -> str:
    return {
        StoreState.PARTIAL: "cohort_bridge_partial",
        StoreState.UNKNOWN: "cohort_bridge_unknown",
        StoreState.CONFLICT: "cohort_bridge_conflict",
        StoreState.UNSUPPORTED: "cohort_bridge_unsupported",
        StoreState.DENIED: "cohort_bridge_denied",
        StoreState.FAILURE: "cohort_bridge_failed",
    }[state]


def _safe_environment() -> dict[str, str]:
    allowed = ("PATH", "LANG", "LC_ALL", "TMPDIR", "SYSTEMROOT", "WINDIR")
    return {key: os.environ[key] for key in allowed if key in os.environ}


__all__ = [
    "COHORT_SERVICE_COMPATIBILITY_POLICY",
    "COHORT_SERVICE_PROTOCOL_VERSION",
    "DEFAULT_COHORT_SERVICE_TIMEOUT_SECONDS",
    "MAX_COHORT_SERVICE_RESPONSE_BYTES",
    "CohortDefinitionServiceBridge",
    "CohortServiceBridgeError",
    "CohortServiceConversion",
    "CohortServiceProtocolError",
    "CohortServiceUnavailableError",
]
