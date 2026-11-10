"""Optional out-of-process bridge for user-supplied ARX adapters.

ARX is JVM-based and is never bundled or started implicitly. This bridge uses
a small JSON-over-stdin protocol so source rows do not appear in process
arguments, logs, or temporary files. A caller supplies an executable adapter
that embeds ARX and implements the documented request/response schema.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from openmed.core.audit import stable_hash

ARX_PROTOCOL_SCHEMA_VERSION: Final = 1
DEFAULT_ARX_TIMEOUT_SECONDS: Final = 300.0


class ArxBridgeError(RuntimeError):
    """Base class for privacy-safe ARX bridge failures."""


class ArxNotAvailableError(ArxBridgeError):
    """Raised when no usable user-supplied ARX adapter is configured."""


class ArxProtocolError(ArxBridgeError):
    """Raised when an ARX adapter returns an invalid response."""


@dataclass(frozen=True)
class ArxResult:
    """Anonymized records and a raw-value-free ARX transformation manifest.

    Attributes:
        records: Released row mappings returned by the adapter.
        manifest: Sanitized engine, target, level, count, and hash evidence.
    """

    records: tuple[dict[str, Any], ...]
    manifest: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ArxBridge:
    """Invoke a caller-provided ARX adapter without a shell or temp files.

    The adapter receives one UTF-8 JSON object on standard input with
    ``schema_version``, ``records``, ``quasi_identifiers``,
    ``sensitive_attributes``, and ``privacy`` fields. It must write one JSON
    object containing a ``records`` list and may include
    ``generalization_levels`` plus ``suppressed_count``. Standard error is
    discarded so a misconfigured adapter cannot echo source cells into logs.

    Attributes:
        command: Executable and arguments for the user-supplied adapter.
        timeout_seconds: Maximum adapter runtime before fail-closed termination.
    """

    command: tuple[str, ...] | None = None
    timeout_seconds: float = DEFAULT_ARX_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        if self.command is not None:
            if not self.command or any(
                not isinstance(part, str) or not part for part in self.command
            ):
                raise ValueError("command must contain non-empty string arguments")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or self.timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be positive")

    @property
    def available(self) -> bool:
        """Return whether the configured adapter executable can be resolved."""

        if self.command is None:
            return False
        executable = self.command[0]
        path = Path(executable)
        if path.is_absolute() or path.parent != Path("."):
            executable_available = path.is_file() and os.access(path, os.X_OK)
        else:
            executable_available = shutil.which(executable) is not None
        if not executable_available:
            return False
        if len(self.command) >= 3 and self.command[1] == "-jar":
            return Path(self.command[2]).is_file()
        return True

    def anonymize(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        quasi_identifiers: Mapping[str, str],
        sensitive_attributes: Sequence[str] = (),
        k: int = 2,
        l: int = 1,
        t: float = 1.0,
    ) -> ArxResult:
        """Anonymize records through the configured user-supplied adapter.

        Args:
            records: Non-empty source row mappings.
            quasi_identifiers: Reviewed column-to-hierarchy mapping.
            sensitive_attributes: Columns used for l/t privacy models.
            k: Minimum equivalence-class size.
            l: Minimum sensitive-value diversity.
            t: Maximum variational t-closeness distance.

        Returns:
            Parsed records and a sanitized bridge manifest.

        Raises:
            ArxNotAvailableError: If the configured executable cannot be used.
            ArxBridgeError: If execution fails or times out.
            ArxProtocolError: If input cannot be encoded or output is invalid.
            ValueError: If the privacy configuration or columns are invalid.
        """

        if not self.available or self.command is None:
            raise ArxNotAvailableError(
                "ARX is not available; configure an executable adapter command "
                "or use the pure-Python engine"
            )
        payload = _request_payload(
            records,
            quasi_identifiers=quasi_identifiers,
            sensitive_attributes=sensitive_attributes,
            k=k,
            l=l,
            t=t,
        )
        try:
            encoded = json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ArxProtocolError("ARX input contains an unsupported value") from exc

        try:
            completed = subprocess.run(
                self.command,
                input=encoded,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=float(self.timeout_seconds),
                env=_safe_environment(),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ArxBridgeError("ARX adapter execution failed") from exc
        if completed.returncode != 0:
            raise ArxBridgeError(
                f"ARX adapter exited unsuccessfully ({completed.returncode})"
            )
        return _parse_response(
            completed.stdout,
            quasi_identifiers=quasi_identifiers,
            k=k,
            l=l,
            t=t,
            source_count=len(records),
        )


def _request_payload(
    records: Sequence[Mapping[str, Any]],
    *,
    quasi_identifiers: Mapping[str, str],
    sensitive_attributes: Sequence[str],
    k: int,
    l: int,
    t: float,
) -> dict[str, Any]:
    if not records or not all(isinstance(row, Mapping) for row in records):
        raise ValueError("records must be a non-empty sequence of mappings")
    if not quasi_identifiers:
        raise ValueError("quasi_identifiers must not be empty")
    if any(column not in row for row in records for column in quasi_identifiers):
        raise ValueError("a quasi-identifier column is absent from the table")
    sensitive = tuple(dict.fromkeys(sensitive_attributes))
    if any(column not in row for row in records for column in sensitive):
        raise ValueError("a sensitive-attribute column is absent from the table")
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    if isinstance(l, bool) or not isinstance(l, int) or l < 1:
        raise ValueError("l must be a positive integer")
    if isinstance(t, bool) or not isinstance(t, (int, float)) or not 0 <= t <= 1:
        raise ValueError("t must be between 0 and 1")
    return {
        "schema_version": ARX_PROTOCOL_SCHEMA_VERSION,
        "records": [dict(row) for row in records],
        "quasi_identifiers": dict(quasi_identifiers),
        "sensitive_attributes": list(sensitive),
        "privacy": {"k": k, "l": l, "t": float(t)},
    }


def _parse_response(
    stdout: bytes,
    *,
    quasi_identifiers: Mapping[str, str],
    k: int,
    l: int,
    t: float,
    source_count: int,
) -> ArxResult:
    try:
        payload = json.loads(stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ArxProtocolError("ARX adapter returned invalid JSON") from None
    if not isinstance(payload, Mapping):
        raise ArxProtocolError("ARX adapter response must be an object")
    raw_records = payload.get("records")
    if not isinstance(raw_records, list) or not all(
        isinstance(row, Mapping) for row in raw_records
    ):
        raise ArxProtocolError("ARX adapter response contains invalid records")
    records = tuple(dict(row) for row in raw_records)
    if not records:
        raise ArxProtocolError("ARX adapter returned an empty release")
    for column in quasi_identifiers:
        if any(column not in row for row in records):
            raise ArxProtocolError(
                "ARX adapter removed a required quasi-identifier column"
            )

    raw_levels = payload.get("generalization_levels", {})
    if not isinstance(raw_levels, Mapping):
        raise ArxProtocolError("ARX generalization levels must be an object")
    levels: dict[str, int] = {}
    for column in quasi_identifiers:
        level = raw_levels.get(column, 0)
        if isinstance(level, bool) or not isinstance(level, int) or level < 0:
            raise ArxProtocolError("ARX returned an invalid generalization level")
        levels[column] = level
    suppressed_count = payload.get("suppressed_count", source_count - len(records))
    if (
        isinstance(suppressed_count, bool)
        or not isinstance(suppressed_count, int)
        or suppressed_count < 0
        or suppressed_count > source_count
        or suppressed_count != source_count - len(records)
    ):
        raise ArxProtocolError("ARX returned an invalid suppression count")

    manifest = {
        "manifest_schema_version": "1.0.0",
        "engine": "arx",
        "protocol_schema_version": ARX_PROTOCOL_SCHEMA_VERSION,
        "target_k": k,
        "target_l": l,
        "target_t": float(t),
        "quasi_identifiers": dict(quasi_identifiers),
        "generalization_levels": levels,
        "record_count": source_count,
        "released_count": len(records),
        "suppressed_count": suppressed_count,
        "output_hash": stable_hash(records),
    }
    return ArxResult(records=records, manifest=manifest)


def _safe_environment() -> dict[str, str]:
    """Return the non-secret process environment an adapter may require."""

    allowed = ("JAVA_HOME", "LANG", "LC_ALL", "PATH", "SYSTEMROOT", "TMPDIR")
    return {name: os.environ[name] for name in allowed if name in os.environ}


__all__ = [
    "ARX_PROTOCOL_SCHEMA_VERSION",
    "DEFAULT_ARX_TIMEOUT_SECONDS",
    "ArxBridge",
    "ArxBridgeError",
    "ArxNotAvailableError",
    "ArxProtocolError",
    "ArxResult",
]
