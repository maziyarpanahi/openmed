"""Optional out-of-process bridge for user-supplied ARX anonymization.

ARX is a JVM application and is intentionally not a Python dependency or a
bundled artifact.  This module invokes a caller-selected ARX runner without a
shell.  The runner is expected to accept the following arguments and write a
JSON result to the supplied output path::

    <runner> --input <records.json> --config <config.json> --output <result.json>

``records.json`` is an array of row mappings.  ``config.json`` contains the
OpenMed quasi-identifier, sensitive-attribute, hierarchy, and target k/l
configuration.  A result may be either an array of anonymized rows or an
object with a ``records`` (or ``anonymized_records``) array and optional
``metadata`` object.  A small wrapper around the user's ARX installation can
therefore implement this stable file contract without adding ARX to OpenMed's
runtime or distribution.
"""

from __future__ import annotations

import json
import math
import os
import shlex
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

__all__ = [
    "ARX_COMMAND_ENV",
    "ARX_PROTOCOL_VERSION",
    "ARXBridge",
    "ARXBridgeError",
    "ARXConfig",
    "ARXAnonymizationResult",
    "ARXNotAvailableError",
    "ARXResult",
    "ARX_UNAVAILABLE_SENTINEL",
    "ARXUnavailableError",
    "ArxBridge",
    "anonymize_with_arx",
    "run_arx",
]

ARX_COMMAND_ENV: Final = "OPENMED_ARX_COMMAND"
ARX_PROTOCOL_VERSION: Final = 1
ARX_UNAVAILABLE_SENTINEL: Final = "OPENMED_ARX_UNAVAILABLE"
_DEFAULT_TIMEOUT: Final = 60.0
_MAX_RESULT_BYTES: Final = 16 * 1024 * 1024
_RESULT_RECORD_KEYS: Final = ("records", "anonymized_records", "data")
_CommandSpec = str | os.PathLike[str] | Sequence[str | os.PathLike[str]]


class ARXBridgeError(RuntimeError):
    """Base exception for fail-closed ARX bridge failures."""


class ARXUnavailableError(ARXBridgeError):
    """Raised when the caller-supplied ARX runner is unavailable."""


# Both spellings are useful to callers and keep the error name readable in
# code that uses the issue's "not available" terminology.
ARXNotAvailableError = ARXUnavailableError


@dataclass(frozen=True)
class ARXConfig:
    """Validated OpenMed configuration passed to an ARX runner.

    ``hierarchies`` maps a quasi-identifier column to a JSON-compatible
    hierarchy definition.  OpenMed's structured hierarchy output (a list of
    level mappings) is accepted, as are caller-defined mapping forms.
    """

    quasi_identifiers: tuple[str, ...]
    sensitive_attributes: tuple[str, ...] = ()
    hierarchies: dict[str, Any] = field(default_factory=dict)
    target_k: int = 2
    target_l: int = 1

    def __post_init__(self) -> None:
        """Normalize and validate configuration at construction time."""

        normalized = _normalize_config(
            self.quasi_identifiers,
            self.sensitive_attributes,
            self.hierarchies,
            self.target_k,
            self.target_l,
        )
        object.__setattr__(self, "quasi_identifiers", normalized[0])
        object.__setattr__(self, "sensitive_attributes", normalized[1])
        object.__setattr__(self, "hierarchies", normalized[2])
        object.__setattr__(self, "target_k", normalized[3])
        object.__setattr__(self, "target_l", normalized[4])

    def to_dict(self) -> dict[str, Any]:
        """Return the deterministic JSON contract sent to the runner."""

        return {
            "schema_version": ARX_PROTOCOL_VERSION,
            "quasi_identifiers": list(self.quasi_identifiers),
            "sensitive_attributes": list(self.sensitive_attributes),
            "hierarchies": _canonical_json_value(self.hierarchies),
            "target_k": self.target_k,
            "target_l": self.target_l,
        }


@dataclass(frozen=True)
class ARXAnonymizationResult:
    """Deterministic anonymized rows and optional aggregate runner metadata."""

    records: tuple[dict[str, Any], ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable result mapping."""

        return {
            "records": [dict(record) for record in self.records],
            "metadata": _canonical_json_value(self.metadata),
        }

    def __getitem__(self, key: str) -> Any:
        """Provide mapping-style access for integrations using result dicts."""

        return self.to_dict()[key]


ARXResult = ARXAnonymizationResult


class ARXBridge:
    """Run a user-supplied ARX wrapper as a bounded subprocess.

    Args:
        arx_command: Executable name/path plus optional fixed arguments.  If
            omitted, ``OPENMED_ARX_COMMAND`` is parsed with :mod:`shlex`.
        timeout: Maximum subprocess runtime in seconds.

    The command is never executed through a shell.  OpenMed does not discover,
    download, or bundle an ARX JAR or a JVM; callers must install and configure
    their own runner.
    """

    def __init__(
        self,
        arx_command: _CommandSpec | None = None,
        *,
        command: _CommandSpec | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        if arx_command is not None and command is not None:
            raise TypeError("provide only one of arx_command or command")
        self._arx_command = arx_command if arx_command is not None else command
        self.timeout = _validate_timeout(timeout)

    @property
    def arx_command(self) -> _CommandSpec | None:
        """Return the configured command, or ``None`` for environment lookup."""

        return self._arx_command

    def is_available(self) -> bool:
        """Return whether the configured runner can be resolved locally."""

        try:
            _resolve_command(self._arx_command)
        except ARXUnavailableError:
            return False
        return True

    @property
    def available(self) -> bool:
        """Whether the configured runner can be resolved locally."""

        return self.is_available()

    def anonymize(
        self,
        records: Sequence[Mapping[str, Any]],
        quasi_identifiers: Sequence[str] | ARXConfig | Mapping[str, Any] | None = None,
        *,
        sensitive_attributes: Sequence[str] = (),
        hierarchies: Mapping[str, Any] | None = None,
        target_k: int = 2,
        target_l: int = 1,
        config: ARXConfig | Mapping[str, Any] | None = None,
    ) -> ARXAnonymizationResult:
        """Anonymize synthetic/local rows through the configured ARX runner.

        ``quasi_identifiers`` may be replaced by an :class:`ARXConfig` (or a
        config mapping) as the second positional argument.  The runner may
        suppress rows, so the result can contain fewer rows than the input.

        Raises:
            ARXUnavailableError: If no executable ARX runner is configured.
            ARXBridgeError: If the subprocess or result contract fails.
            TypeError, ValueError: If records or privacy configuration is
                invalid.
        """

        resolved_command = _resolve_command(self._arx_command)
        config = _coerce_config(
            quasi_identifiers,
            sensitive_attributes=sensitive_attributes,
            hierarchies=hierarchies,
            target_k=target_k,
            target_l=target_l,
            config=config,
        )
        rows = _materialize_records(records)
        _validate_selected_columns(rows, config)
        return _run_arx(
            rows,
            config=config,
            command=resolved_command,
            timeout=self.timeout,
        )

    run = anonymize


ArxBridge = ARXBridge


def run_arx(
    records: Sequence[Mapping[str, Any]],
    *,
    quasi_identifiers: Sequence[str],
    sensitive_attributes: Sequence[str] = (),
    hierarchies: Mapping[str, Any] | None = None,
    target_k: int = 2,
    target_l: int = 1,
    arx_command: _CommandSpec | None = None,
    command: _CommandSpec | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
) -> ARXAnonymizationResult:
    """Run the configured ARX bridge for one local table."""

    return ARXBridge(
        arx_command,
        command=command,
        timeout=timeout,
    ).anonymize(
        records,
        quasi_identifiers,
        sensitive_attributes=sensitive_attributes,
        hierarchies=hierarchies,
        target_k=target_k,
        target_l=target_l,
    )


anonymize_with_arx = run_arx


def _coerce_config(
    quasi_identifiers: Sequence[str] | ARXConfig | Mapping[str, Any] | None,
    *,
    sensitive_attributes: Sequence[str],
    hierarchies: Mapping[str, Any] | None,
    target_k: int,
    target_l: int,
    config: ARXConfig | Mapping[str, Any] | None,
) -> ARXConfig:
    if config is not None and quasi_identifiers is not None:
        raise TypeError("provide privacy settings either in config or as arguments")
    if config is None and isinstance(quasi_identifiers, (ARXConfig, Mapping)):
        config = quasi_identifiers
        quasi_identifiers = None
    if config is not None:
        if isinstance(config, ARXConfig):
            return config
        if not isinstance(config, Mapping):
            raise TypeError("config must be an ARXConfig or mapping")
        try:
            return ARXConfig(
                quasi_identifiers=config["quasi_identifiers"],
                sensitive_attributes=config.get("sensitive_attributes", ()),
                hierarchies=config.get("hierarchies"),
                target_k=config.get("target_k", 2),
                target_l=config.get("target_l", 1),
            )
        except KeyError as exc:
            raise ValueError("config must contain quasi_identifiers") from exc
    if quasi_identifiers is None:
        raise TypeError("quasi_identifiers must be provided")
    return ARXConfig(
        quasi_identifiers=quasi_identifiers,
        sensitive_attributes=sensitive_attributes,
        hierarchies=hierarchies or {},
        target_k=target_k,
        target_l=target_l,
    )


def _normalize_config(
    quasi_identifiers: Sequence[str],
    sensitive_attributes: Sequence[str],
    hierarchies: Mapping[str, Any] | None,
    target_k: int,
    target_l: int,
) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, Any], int, int]:
    quasi = _normalize_columns(
        quasi_identifiers,
        label="quasi_identifiers",
        allow_empty=False,
    )
    sensitive = _normalize_columns(
        sensitive_attributes,
        label="sensitive_attributes",
        allow_empty=True,
    )
    overlap = sorted(set(quasi) & set(sensitive))
    if overlap:
        raise ValueError(
            "sensitive_attributes must not also be quasi_identifiers: "
            + ", ".join(overlap)
        )
    normalized_hierarchies = _normalize_hierarchies(hierarchies, quasi)
    normalized_k = _validate_positive_integer(target_k, "target_k")
    normalized_l = _validate_positive_integer(target_l, "target_l")
    if normalized_l > 1 and not sensitive:
        raise ValueError("target_l > 1 requires at least one sensitive attribute")
    return quasi, sensitive, normalized_hierarchies, normalized_k, normalized_l


def _normalize_columns(
    columns: Sequence[str],
    *,
    label: str,
    allow_empty: bool,
) -> tuple[str, ...]:
    if isinstance(columns, (str, bytes, bytearray)) or not isinstance(
        columns, Sequence
    ):
        raise TypeError(f"{label} must be a sequence of column names")
    normalized: list[str] = []
    for column in columns:
        if not isinstance(column, str):
            raise TypeError(f"{label} entries must be strings")
        if not column.strip() or "\x00" in column:
            raise ValueError(f"{label} entries must be non-empty column names")
        normalized.append(column)
    if not normalized and not allow_empty:
        raise ValueError(f"{label} must contain at least one column")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{label} must not contain duplicate columns")
    return tuple(normalized)


def _normalize_hierarchies(
    hierarchies: Mapping[str, Any] | None,
    quasi_identifiers: Sequence[str],
) -> dict[str, Any]:
    if hierarchies is None:
        return {}
    if not isinstance(hierarchies, Mapping):
        raise TypeError("hierarchies must be a mapping of column names to definitions")
    normalized: dict[str, Any] = {}
    for column, definition in hierarchies.items():
        if not isinstance(column, str) or not column.strip() or "\x00" in column:
            raise ValueError("hierarchy column names must be non-empty strings")
        if column not in quasi_identifiers:
            raise ValueError(
                f"hierarchy is declared for non-quasi-identifier column {column!r}"
            )
        normalized[column] = _canonical_json_value(definition)
    return normalized


def _validate_positive_integer(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be an integer >= 1")
    return value


def _validate_timeout(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("timeout must be a positive finite number")
    timeout = float(value)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be a positive finite number")
    return timeout


def _materialize_records(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if isinstance(records, (str, bytes, bytearray)) or not isinstance(
        records, Sequence
    ):
        raise TypeError("records must be a sequence of row mappings")
    if not records:
        raise ValueError("records must contain at least one row")
    materialized: list[dict[str, Any]] = []
    for row_index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(f"records[{row_index}] must be a row mapping")
        row: dict[str, Any] = {}
        for field_name, value in record.items():
            if (
                not isinstance(field_name, str)
                or not field_name.strip()
                or "\x00" in field_name
            ):
                raise ValueError(
                    f"records[{row_index}] contains an invalid column name"
                )
            row[field_name] = _validate_scalar(
                value,
                label=f"records[{row_index}][{field_name!r}]",
            )
        materialized.append(row)
    return materialized


def _validate_scalar(value: Any, *, label: str) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} must be finite when it is a float")
        return value
    raise TypeError(f"{label} must be a scalar value or None")


def _validate_selected_columns(
    records: Sequence[Mapping[str, Any]],
    config: ARXConfig,
) -> None:
    available = {field for record in records for field in record}
    selected = set(config.quasi_identifiers) | set(config.sensitive_attributes)
    missing = sorted(selected - available)
    if missing:
        raise ValueError(
            "configured ARX columns are absent from records: " + ", ".join(missing)
        )


def _resolve_command(command: _CommandSpec | None) -> list[str]:
    if command is None:
        configured = os.environ.get(ARX_COMMAND_ENV)
        if not configured:
            raise ARXUnavailableError(_not_available_message())
        command = configured
    parts = _command_parts(command)
    executable = shutil.which(parts[0])
    if executable is None:
        raise ARXUnavailableError(
            f"ARX runner {parts[0]!r} is unavailable. " + _not_available_message()
        )
    return [os.path.abspath(executable), *parts[1:]]


def _command_parts(command: _CommandSpec) -> list[str]:
    if isinstance(command, os.PathLike):
        parts = [os.fspath(command)]
    elif isinstance(command, str):
        try:
            parts = shlex.split(command)
        except ValueError as exc:
            raise ValueError(
                "arx_command is not valid shell-like command text"
            ) from exc
    elif isinstance(command, Sequence) and not isinstance(command, (bytes, bytearray)):
        parts = [
            os.fspath(part) if isinstance(part, os.PathLike) else part
            for part in command
        ]
    else:
        raise TypeError("arx_command must be a command string, path, or sequence")
    if not parts or any(not isinstance(part, str) or not part for part in parts):
        raise ValueError("arx_command must contain a non-empty executable")
    if any("\x00" in part for part in parts):
        raise ValueError("arx_command must not contain NUL characters")
    return parts


def _not_available_message() -> str:
    return (
        "Install ARX and provide an executable wrapper with arx_command or set "
        f"{ARX_COMMAND_ENV}; OpenMed does not bundle the ARX JAR or a JVM. The "
        "wrapper must accept --input, --config, and --output paths."
    )


def _run_arx(
    records: Sequence[Mapping[str, Any]],
    *,
    config: ARXConfig,
    command: Sequence[str],
    timeout: float,
) -> ARXAnonymizationResult:
    with tempfile.TemporaryDirectory(prefix="openmed-arx-") as directory:
        workspace = Path(directory)
        input_path = workspace / "records.json"
        config_path = workspace / "config.json"
        output_path = workspace / "result.json"
        _write_private_json(input_path, records)
        _write_private_json(config_path, config.to_dict())
        _create_private_file(output_path)

        invocation = [
            *command,
            "--input",
            os.fspath(input_path),
            "--config",
            os.fspath(config_path),
            "--output",
            os.fspath(output_path),
        ]
        try:
            completed = subprocess.run(  # noqa: S603 - explicit executable/argv
                invocation,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise ARXBridgeError(
                f"ARX runner timed out after {timeout:g} seconds"
            ) from exc
        except (FileNotFoundError, PermissionError, OSError) as exc:
            raise ARXUnavailableError(
                "The configured ARX runner could not be started. "
                + _not_available_message()
            ) from exc

        stdout = completed.stdout if isinstance(completed.stdout, str) else ""
        stderr = completed.stderr if isinstance(completed.stderr, str) else ""
        if completed.returncode != 0:
            if ARX_UNAVAILABLE_SENTINEL in f"{stdout}\n{stderr}":
                raise ARXUnavailableError(
                    "The ARX installation is unavailable to the configured runner. "
                    + _not_available_message()
                )
            raise ARXBridgeError(
                "ARX runner failed with exit status "
                f"{completed.returncode}; no result was accepted"
            )

        payload = _read_result_payload(output_path, stdout)
    try:
        return _parse_result(payload)
    except (TypeError, ValueError) as exc:
        raise ARXBridgeError(
            "ARX runner produced an invalid anonymized result"
        ) from exc


def _write_private_json(path: Path, value: Any) -> None:
    encoded = json.dumps(
        _canonical_json_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
        handle.write(encoded)


def _create_private_file(path: Path) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)


def _read_result_payload(path: Path, stdout: str) -> Any:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ARXBridgeError(
            "ARX runner did not produce a result JSON payload"
        ) from exc
    if size > _MAX_RESULT_BYTES:
        raise ARXBridgeError("ARX result JSON exceeds the permitted size")
    raw = path.read_text(encoding="utf-8") if size else stdout.strip()
    if not raw:
        raise ARXBridgeError("ARX runner did not produce a result JSON payload")
    try:
        return json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ARXBridgeError("ARX runner produced invalid result JSON") from exc


def _parse_result(payload: Any) -> ARXAnonymizationResult:
    if isinstance(payload, list):
        raw_records = payload
        raw_metadata: Any = {}
    elif isinstance(payload, Mapping):
        record_key = next(
            (key for key in _RESULT_RECORD_KEYS if key in payload),
            None,
        )
        if record_key is None:
            raise ARXBridgeError(
                "ARX result must contain a records or anonymized_records array"
            )
        raw_records = payload[record_key]
        raw_metadata = payload.get("metadata", {})
        if raw_metadata == {}:
            raw_metadata = {
                key: value
                for key, value in payload.items()
                if key not in _RESULT_RECORD_KEYS
            }
    else:
        raise ARXBridgeError("ARX result must be a JSON array or object")

    if not isinstance(raw_records, list):
        raise ARXBridgeError("ARX result records must be a JSON array")
    records = tuple(_materialize_records(raw_records))
    if not isinstance(raw_metadata, Mapping):
        raise ARXBridgeError("ARX result metadata must be a JSON object")
    metadata = _canonical_json_value(raw_metadata)
    return ARXAnonymizationResult(records=records, metadata=metadata)


def _canonical_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON values must contain only finite numbers")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key or "\x00" in key:
                raise TypeError("JSON object keys must be non-empty strings")
            normalized[key] = _canonical_json_value(item)
        return {key: normalized[key] for key in sorted(normalized)}
    if isinstance(value, (list, tuple)):
        return [_canonical_json_value(item) for item in value]
    raise TypeError("hierarchies and metadata must contain JSON-compatible values")
