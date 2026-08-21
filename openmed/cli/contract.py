"""Deterministic, value-free fixtures for the CLI machine contract.

The fixtures exercise only the shared JSON envelope helpers.  They deliberately
do not call a model, read a user file, or resolve a remote resource.  This keeps
the contract gate useful in release automation even when the local model cache
is empty or the network is unavailable.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from io import StringIO
from typing import Any

from ._output import (
    EXIT_ERROR,
    EXIT_OK,
    EXIT_USAGE,
    CliError,
    emit,
    emit_error,
)

VALIDATION_ERROR_CODE = "invalid_discovery_config"
VALIDATION_ERROR_MESSAGE = (
    "The structured discovery configuration does not match the input schema."
)
OFFLINE_ERROR_CODE = "offline_unavailable"
OFFLINE_ERROR_MESSAGE = "The requested operation requires a local model cache."
PRIVACY_POLICY_ERROR_CODE = "release_policy_failed"
PRIVACY_POLICY_ERROR_MESSAGE = (
    "Structured release does not meet the configured privacy policy; "
    "the aggregate assessment was written to the requested output."
)


@dataclass(frozen=True)
class CliContractFixture:
    """One synthetic command outcome pinned by the machine contract.

    A fixture contains only command metadata and an aggregate result.  It never
    stores command input, paths, exception details, or record-level values.

    Args:
        name: Stable scenario identifier used by the test gate.
        command: Space-separated command path exposed in the JSON envelope.
        expected_exit_code: Process status expected for this scenario.
        data: Metadata-only success payload, or ``None`` for an error case.
        error_code: Stable error category for an error case.
        error_message: Value-free error message for an error case.
    """

    name: str
    command: str
    expected_exit_code: int
    data: dict[str, Any] | None = None
    error_code: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        if not self.name or self.name != self.name.strip():
            raise ValueError("fixture names must be non-empty and trimmed")
        if not self.command or self.command != self.command.strip():
            raise ValueError("command paths must be non-empty and trimmed")
        if self.expected_exit_code not in {EXIT_OK, EXIT_ERROR, EXIT_USAGE}:
            raise ValueError("fixtures must use one of the documented exit codes")

        has_data = self.data is not None
        has_error = self.error_code is not None or self.error_message is not None
        if has_data == has_error:
            raise ValueError("a fixture must describe either success or failure")

        if self.expected_exit_code == EXIT_OK and not has_data:
            raise ValueError("successful fixtures must provide data")
        if self.expected_exit_code != EXIT_OK and not has_error:
            raise ValueError("failed fixtures must provide an error")

        if self.data is not None:
            _copy_json_object(self.data)
        else:
            if not self.error_code or not self.error_code.strip():
                raise ValueError("failed fixtures must provide an error code")
            if not self.error_message or not self.error_message.strip():
                raise ValueError("failed fixtures must provide an error message")
            if any(character in self.error_message for character in "\r\n"):
                raise ValueError("fixture error messages must be single-line")

    @property
    def expected_payload(self) -> dict[str, Any]:
        """Return the exact JSON envelope expected for this fixture."""

        if self.data is not None:
            return {
                "ok": True,
                "command": self.command,
                "data": _copy_json_object(self.data),
            }
        return {
            "ok": False,
            "command": self.command,
            "error": {
                "code": self.error_code,
                "message": self.error_message,
            },
        }


@dataclass(frozen=True)
class CliContractResult:
    """Rendered result returned by :func:`render_contract_fixture`."""

    exit_code: int
    payload: dict[str, Any]
    json_text: str


def render_contract_fixture(fixture: CliContractFixture) -> CliContractResult:
    """Render one fixture through the production JSON envelope helpers.

    This function is intentionally local-only.  Its only side effect is
    writing one JSON document to an in-memory stream.
    """

    if not isinstance(fixture, CliContractFixture):
        raise TypeError("fixture must be a CliContractFixture")

    args = argparse.Namespace(json_output=True, command_path=fixture.command)
    stream = StringIO()
    if fixture.data is not None:
        exit_code = emit(args, _copy_json_object(fixture.data), stream=stream)
    else:
        error = CliError(
            fixture.error_message or "Contract fixture failed.",
            code=fixture.error_code or "error",
            exit_code=fixture.expected_exit_code,
        )
        exit_code = emit_error(args, error, json_stream=stream)

    json_text = stream.getvalue()
    payload = json.loads(json_text)
    if not isinstance(payload, dict):  # pragma: no cover - guarded by helpers
        raise ValueError("CLI contract output must be a JSON object")
    return CliContractResult(
        exit_code=exit_code,
        payload=payload,
        json_text=json_text,
    )


def _copy_json_object(value: dict[str, Any]) -> dict[str, Any]:
    try:
        serialized = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
        )
        restored = json.loads(serialized)
    except (TypeError, ValueError):
        raise ValueError("fixture data must be a finite JSON object") from None
    if type(restored) is not dict:
        raise ValueError("fixture data must be a finite JSON object")
    return restored


SUCCESS_FIXTURE = CliContractFixture(
    name="success",
    command="models list",
    expected_exit_code=EXIT_OK,
    data={"count": 0, "models": []},
)

VALIDATION_FAILURE_FIXTURE = CliContractFixture(
    name="validation",
    command="risk discover",
    expected_exit_code=EXIT_USAGE,
    error_code=VALIDATION_ERROR_CODE,
    error_message=VALIDATION_ERROR_MESSAGE,
)

OFFLINE_FAILURE_FIXTURE = CliContractFixture(
    name="offline",
    command="models pull",
    expected_exit_code=EXIT_ERROR,
    error_code=OFFLINE_ERROR_CODE,
    error_message=OFFLINE_ERROR_MESSAGE,
)

PRIVACY_POLICY_FAILURE_FIXTURE = CliContractFixture(
    name="privacy_policy",
    command="risk assess",
    expected_exit_code=EXIT_ERROR,
    error_code=PRIVACY_POLICY_ERROR_CODE,
    error_message=PRIVACY_POLICY_ERROR_MESSAGE,
)

CONTRACT_FIXTURES = (
    SUCCESS_FIXTURE,
    VALIDATION_FAILURE_FIXTURE,
    OFFLINE_FAILURE_FIXTURE,
    PRIVACY_POLICY_FAILURE_FIXTURE,
)


__all__ = [
    "CONTRACT_FIXTURES",
    "OFFLINE_ERROR_CODE",
    "OFFLINE_ERROR_MESSAGE",
    "OFFLINE_FAILURE_FIXTURE",
    "PRIVACY_POLICY_ERROR_CODE",
    "PRIVACY_POLICY_ERROR_MESSAGE",
    "PRIVACY_POLICY_FAILURE_FIXTURE",
    "SUCCESS_FIXTURE",
    "VALIDATION_ERROR_CODE",
    "VALIDATION_ERROR_MESSAGE",
    "VALIDATION_FAILURE_FIXTURE",
    "CliContractFixture",
    "CliContractResult",
    "render_contract_fixture",
]
