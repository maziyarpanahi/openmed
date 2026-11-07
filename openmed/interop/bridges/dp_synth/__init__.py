"""Strict subprocess contract for permissively licensed DP synthesizers.

The bridge never imports an engine package and never sends source rows or a
source path to the child process.  Engines receive only a declared column
schema and aggregate statistics through JSON on standard input.  A capability
handshake verifies the protocol, aggregate-only boundary, engine family, and
SPDX license before privacy budget is spent or aggregate statistics are sent.

Protocol version 1 has two operations:

``capabilities``
    Returns engine identity plus ``accepts_raw_rows: false`` and the
    ``aggregate-statistics-only`` input contract.
``fit_synthesize``
    Receives ``schema``, ``statistics``, ``privacy``, ``row_count``, and
    ``seed`` and returns synthetic ``rows`` plus the exact privacy spend.

The executable is supplied by the user and runs out of process.  OpenMed does
not bundle, link, or dynamically import it.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

DP_SYNTH_PROTOCOL_VERSION = 1
DEFAULT_DP_SYNTH_EXECUTABLE = "openmed-dp-synth-engine"
DEFAULT_ENGINE_TIMEOUT_SECONDS = 120.0
MAX_ENGINE_RESPONSE_BYTES = 64 * 1024 * 1024

PERMISSIVE_ENGINE_LICENSES = frozenset(
    {
        "Apache-2.0",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "ISC",
        "MIT",
    }
)
SUPPORTED_ENGINE_FAMILIES = frozenset({"dp-gan", "graphical", "marginal"})
SUPPORTED_COLUMN_KINDS = frozenset({"boolean", "integer", "number", "string"})

_FORBIDDEN_STATISTIC_KEYS = frozenset(
    {"data", "raw_rows", "records", "rows", "source_path", "source_rows"}
)


class DPSynthBridgeError(RuntimeError):
    """Base class for safe DP synthesizer bridge failures."""


class DPSynthEngineUnavailable(DPSynthBridgeError):
    """Raised when the optional out-of-process engine cannot be executed."""


class DPSynthProtocolError(DPSynthBridgeError):
    """Raised when an engine violates the aggregate-only JSON protocol."""


class DPSynthLicenseError(DPSynthBridgeError):
    """Raised when an engine does not declare an approved permissive license."""


@dataclass(frozen=True)
class DPSynthEngineInfo:
    """Validated identity and capability metadata for one engine executable."""

    name: str
    version: str
    license: str
    family: str
    protocol_version: int = DP_SYNTH_PROTOCOL_VERSION

    def to_dict(self) -> dict[str, str | int]:
        """Return JSON-compatible engine provenance."""

        return {
            "name": self.name,
            "version": self.version,
            "license": self.license,
            "family": self.family,
            "protocol_version": self.protocol_version,
        }


@dataclass(frozen=True)
class DPSynthResponse:
    """Validated rows and privacy spend returned by a DP engine."""

    rows: tuple[dict[str, Any], ...]
    engine: DPSynthEngineInfo
    epsilon_spent: float
    delta_spent: float


class DPSynthBridge:
    """Invoke one aggregate-only DP synthesizer through a strict subprocess API.

    Args:
        command: Executable path or an argument sequence. A string is treated
            as one executable path and is never evaluated by a shell.
        timeout_seconds: Per-operation subprocess timeout.
    """

    def __init__(
        self,
        command: str | Sequence[str] | None = None,
        *,
        timeout_seconds: float = DEFAULT_ENGINE_TIMEOUT_SECONDS,
    ) -> None:
        self._command = _normalize_command(command)
        self._timeout_seconds = _positive_timeout(timeout_seconds)
        self._resolved_command: tuple[str, ...] | None = None
        self._engine_info: DPSynthEngineInfo | None = None

    @property
    def command(self) -> tuple[str, ...]:
        """Return the configured non-shell command without executing it."""

        return self._command

    def capabilities(self) -> DPSynthEngineInfo:
        """Validate and return aggregate-only engine capabilities.

        The handshake contains no source-derived statistics and is cached for
        the lifetime of this bridge instance.
        """

        if self._engine_info is not None:
            return self._engine_info
        payload = {
            "protocol_version": DP_SYNTH_PROTOCOL_VERSION,
            "operation": "capabilities",
        }
        response = self._invoke(payload)
        _require_protocol_version(response)
        engine_payload = _required_mapping(response, "engine")
        capabilities_payload = _required_mapping(response, "capabilities")
        if capabilities_payload.get("input_contract") != ("aggregate-statistics-only"):
            raise DPSynthProtocolError(
                "DP synthesizer must declare the aggregate-statistics-only "
                "input contract"
            )
        if capabilities_payload.get("accepts_raw_rows") is not False:
            raise DPSynthProtocolError(
                "DP synthesizer must explicitly reject raw source rows"
            )

        license_id = _required_string(engine_payload, "license")
        if license_id not in PERMISSIVE_ENGINE_LICENSES:
            raise DPSynthLicenseError(
                "DP synthesizer license is not on the approved permissive SPDX "
                "allowlist"
            )
        family = _required_string(engine_payload, "family")
        if family not in SUPPORTED_ENGINE_FAMILIES:
            raise DPSynthProtocolError(
                "DP synthesizer family must be marginal, graphical, or dp-gan"
            )
        self._engine_info = DPSynthEngineInfo(
            name=_required_string(engine_payload, "name"),
            version=_required_string(engine_payload, "version"),
            license=license_id,
            family=family,
        )
        return self._engine_info

    def fit_synthesize(
        self,
        schema: Sequence[Mapping[str, Any]],
        stats: Mapping[str, Any],
        *,
        epsilon: float,
        delta: float,
        row_count: int,
        seed: int = 0,
    ) -> DPSynthResponse:
        """Fit from aggregates and return schema-validated synthetic rows.

        Args:
            schema: Ordered column definitions containing only ``name``,
                ``kind``, and ``nullable``.
            stats: JSON-compatible aggregate statistics. Raw-row-shaped keys
                such as ``rows`` and ``records`` are rejected recursively.
            epsilon: Privacy epsilon charged for this generation.
            delta: Privacy delta charged for this generation.
            row_count: Exact number of rows requested from the engine.
            seed: Non-negative deterministic engine seed.

        Returns:
            Validated rows, engine provenance, and declared privacy spend.
        """

        engine = self.capabilities()
        normalized_schema = _validate_schema(schema)
        normalized_stats = _validate_statistics(stats)
        normalized_epsilon = _positive_number(epsilon, field_name="epsilon")
        normalized_delta = _delta_number(delta)
        normalized_rows = _non_negative_integer(row_count, field_name="row_count")
        normalized_seed = _non_negative_integer(seed, field_name="seed")
        if normalized_rows == 0:
            raise ValueError("row_count must be positive")

        payload = {
            "protocol_version": DP_SYNTH_PROTOCOL_VERSION,
            "operation": "fit_synthesize",
            "schema": normalized_schema,
            "statistics": normalized_stats,
            "privacy": {
                "epsilon": normalized_epsilon,
                "delta": normalized_delta,
            },
            "row_count": normalized_rows,
            "seed": normalized_seed,
        }
        response = self._invoke(payload)
        _require_protocol_version(response)
        rows = _validate_rows(
            response.get("rows"),
            normalized_schema,
            expected_count=normalized_rows,
        )
        privacy = _required_mapping(response, "privacy")
        epsilon_spent = _non_negative_number(
            privacy.get("epsilon_spent"),
            field_name="epsilon_spent",
        )
        delta_spent = _delta_number(
            privacy.get("delta_spent"),
            field_name="delta_spent",
        )
        if not math.isclose(
            epsilon_spent,
            normalized_epsilon,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ) or not math.isclose(
            delta_spent,
            normalized_delta,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise DPSynthProtocolError(
                "DP synthesizer reported a privacy spend different from the "
                "budgeted request"
            )
        return DPSynthResponse(
            rows=rows,
            engine=engine,
            epsilon_spent=epsilon_spent,
            delta_spent=delta_spent,
        )

    def _invoke(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        command = self._resolve_command()
        encoded = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        try:
            completed = subprocess.run(
                command,
                check=False,
                env=_minimal_subprocess_environment(),
                input=encoded,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=self._timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise DPSynthBridgeError(
                "DP synthesizer timed out without releasing an output"
            ) from exc
        except OSError as exc:
            raise DPSynthEngineUnavailable(_unavailable_message(command[0])) from exc
        if completed.returncode != 0:
            raise DPSynthBridgeError(
                "DP synthesizer exited unsuccessfully; stderr was suppressed "
                "to prevent source-derived data from entering logs"
            )
        if len(completed.stdout.encode("utf-8")) > MAX_ENGINE_RESPONSE_BYTES:
            raise DPSynthProtocolError("DP synthesizer response exceeds the size limit")
        return _strict_json_object(completed.stdout)

    def _resolve_command(self) -> tuple[str, ...]:
        if self._resolved_command is not None:
            return self._resolved_command
        executable = shutil.which(self._command[0])
        if executable is None:
            raise DPSynthEngineUnavailable(_unavailable_message(self._command[0]))
        self._resolved_command = (executable, *self._command[1:])
        return self._resolved_command


def fit_synthesize(
    schema: Sequence[Mapping[str, Any]],
    stats: Mapping[str, Any],
    *,
    epsilon: float,
    delta: float,
    row_count: int,
    command: str | Sequence[str] | None = None,
    seed: int = 0,
    timeout_seconds: float = DEFAULT_ENGINE_TIMEOUT_SECONDS,
) -> DPSynthResponse:
    """Run the public aggregate-only ``fit_synthesize(schema, stats)`` contract."""

    return DPSynthBridge(command, timeout_seconds=timeout_seconds).fit_synthesize(
        schema,
        stats,
        epsilon=epsilon,
        delta=delta,
        row_count=row_count,
        seed=seed,
    )


def _normalize_command(command: str | Sequence[str] | None) -> tuple[str, ...]:
    if command is None:
        return (DEFAULT_DP_SYNTH_EXECUTABLE,)
    if isinstance(command, str):
        values = (command,)
    elif isinstance(command, Sequence) and not isinstance(command, (bytes, bytearray)):
        values = tuple(command)
    else:
        raise TypeError("command must be an executable path or string sequence")
    if not values or any(not isinstance(value, str) or not value for value in values):
        raise ValueError("command must contain non-empty string arguments")
    return values


def _positive_timeout(value: Any) -> float:
    timeout = _positive_number(value, field_name="timeout_seconds")
    return float(timeout)


def _positive_number(value: Any, *, field_name: str) -> float:
    number = _non_negative_number(value, field_name=field_name)
    if number <= 0.0:
        raise ValueError(f"{field_name} must be positive")
    return number


def _non_negative_number(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite")
    if number < 0.0:
        raise ValueError(f"{field_name} must be non-negative")
    return number


def _delta_number(value: Any, *, field_name: str = "delta") -> float:
    number = _non_negative_number(value, field_name=field_name)
    if number >= 1.0:
        raise ValueError(f"{field_name} must be smaller than 1")
    return number


def _non_negative_integer(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field_name} must be an integer")
    integer = int(value)
    if integer < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return integer


def _validate_schema(
    schema: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(schema, Sequence) or isinstance(schema, (str, bytes, bytearray)):
        raise TypeError("schema must be a sequence of column mappings")
    if not schema:
        raise ValueError("schema must contain at least one column")
    normalized: list[dict[str, Any]] = []
    names: set[str] = set()
    for column in schema:
        if not isinstance(column, Mapping):
            raise TypeError("every schema column must be a mapping")
        if set(column) != {"kind", "name", "nullable"}:
            raise DPSynthProtocolError(
                "schema columns must contain exactly name, kind, and nullable"
            )
        name = _required_string(column, "name")
        if name in names:
            raise ValueError("schema column names must be unique")
        names.add(name)
        kind = _required_string(column, "kind")
        if kind not in SUPPORTED_COLUMN_KINDS:
            raise ValueError(f"unsupported schema kind for column {name!r}")
        nullable = column["nullable"]
        if not isinstance(nullable, bool):
            raise TypeError("schema nullable values must be boolean")
        normalized.append({"name": name, "kind": kind, "nullable": nullable})
    return normalized


def _validate_statistics(stats: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(stats, Mapping):
        raise TypeError("stats must be an aggregate-statistics mapping")
    normalized = _json_value(stats, path="statistics")
    if not isinstance(normalized, dict):  # pragma: no cover - mapping branch above
        raise TypeError("stats must be an aggregate-statistics mapping")
    return normalized


def _json_value(value: Any, *, path: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise TypeError(f"{path} keys must be non-empty strings")
            if key.casefold() in _FORBIDDEN_STATISTIC_KEYS:
                raise DPSynthProtocolError(
                    "aggregate statistics must not contain raw-row-shaped keys"
                )
            result[key] = _json_value(item, path=f"{path}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item, path=f"{path}[]") for item in value]
    raise TypeError(f"{path} contains a non-JSON value")


def _validate_rows(
    value: Any,
    schema: Sequence[Mapping[str, Any]],
    *,
    expected_count: int,
) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list):
        raise DPSynthProtocolError("DP synthesizer rows must be a JSON array")
    if len(value) != expected_count:
        raise DPSynthProtocolError(
            "DP synthesizer returned a row count different from the request"
        )
    names = tuple(str(column["name"]) for column in schema)
    expected_names = set(names)
    rows: list[dict[str, Any]] = []
    for row in value:
        if not isinstance(row, Mapping) or set(row) != expected_names:
            raise DPSynthProtocolError(
                "every synthetic row must contain exactly the declared schema"
            )
        normalized: dict[str, Any] = {}
        for column in schema:
            name = str(column["name"])
            cell = row[name]
            _validate_cell(cell, column)
            normalized[name] = cell
        rows.append(normalized)
    return tuple(rows)


def _validate_cell(value: Any, column: Mapping[str, Any]) -> None:
    if value is None:
        if column["nullable"]:
            return
        raise DPSynthProtocolError("non-nullable synthetic column returned null")
    kind = column["kind"]
    valid = (
        (kind == "boolean" and isinstance(value, bool))
        or (
            kind == "integer" and isinstance(value, int) and not isinstance(value, bool)
        )
        or (
            kind == "number"
            and isinstance(value, Real)
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        )
        or (kind == "string" and isinstance(value, str))
    )
    if not valid:
        raise DPSynthProtocolError(
            "synthetic row cell does not match its declared column kind"
        )


def _required_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise DPSynthProtocolError(f"DP synthesizer response requires {key!r}")
    return value


def _required_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DPSynthProtocolError(f"DP synthesizer response requires {key!r}")
    return value


def _require_protocol_version(payload: Mapping[str, Any]) -> None:
    if payload.get("protocol_version") != DP_SYNTH_PROTOCOL_VERSION:
        raise DPSynthProtocolError("DP synthesizer protocol version mismatch")


def _strict_json_object(payload: str) -> dict[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DPSynthProtocolError(
                    "DP synthesizer response contains duplicate JSON keys"
                )
            result[key] = value
        return result

    def reject_constant(_value: str) -> None:
        raise DPSynthProtocolError(
            "DP synthesizer response contains a non-finite number"
        )

    try:
        decoded = json.loads(
            payload,
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except DPSynthProtocolError:
        raise
    except (json.JSONDecodeError, RecursionError) as exc:
        raise DPSynthProtocolError("DP synthesizer returned invalid JSON") from exc
    if not isinstance(decoded, dict):
        raise DPSynthProtocolError("DP synthesizer response must be a JSON object")
    return decoded


def _minimal_subprocess_environment() -> dict[str, str]:
    allowed = ("LANG", "LC_ALL", "PATH", "SYSTEMROOT", "TMPDIR", "WINDIR")
    return {key: os.environ[key] for key in allowed if key in os.environ}


def _unavailable_message(executable: str) -> str:
    return (
        f"Optional DP synthesizer executable {executable!r} was not found. "
        "Install a separate Apache-2.0, BSD, ISC, or MIT licensed marginal, "
        "graphical, or DP-GAN engine and pass its executable as engine_command."
    )


__all__ = [
    "DEFAULT_DP_SYNTH_EXECUTABLE",
    "DEFAULT_ENGINE_TIMEOUT_SECONDS",
    "DP_SYNTH_PROTOCOL_VERSION",
    "PERMISSIVE_ENGINE_LICENSES",
    "SUPPORTED_COLUMN_KINDS",
    "SUPPORTED_ENGINE_FAMILIES",
    "DPSynthBridge",
    "DPSynthBridgeError",
    "DPSynthEngineInfo",
    "DPSynthEngineUnavailable",
    "DPSynthLicenseError",
    "DPSynthProtocolError",
    "DPSynthResponse",
    "fit_synthesize",
]
