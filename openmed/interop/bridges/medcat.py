"""Subprocess bridge to a user-provided MedCAT/CogStack process.

MedCAT is distributed under the Elastic License 2.0 — a source-available
license that is not OSI-approved open source. Per invariant I2, OpenMed never
imports or bundles MedCAT in-process. This module only shells out to a
MedCAT process or CLI that the caller has installed and configured
themselves, reads its concept output from stdout, and maps
``{cui, name, score}`` records onto OpenMed span-code fields
(``{system, code, score}``).

Invocation is blocked until the caller explicitly acknowledges the Elastic
License 2.0 terms, either by setting ``OPENMED_ACCEPT_MEDCAT_LICENSE`` or by
accepting an interactive terminal prompt.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any, Callable, Mapping, Optional, Sequence

LICENSE_ENV_VAR = "OPENMED_ACCEPT_MEDCAT_LICENSE"

LICENSE_NOTICE = (
    "MedCAT/CogStack is distributed under the Elastic License 2.0, a "
    "source-available license that is not OSI-approved open source. "
    "OpenMed never bundles or imports MedCAT in-process — you are solely "
    "responsible for installing MedCAT and accepting its license directly "
    "with CogStack/Elastic. See https://github.com/CogStack/MedCAT for the "
    "full license text."
)

DEFAULT_SYSTEM = "UMLS"
_VALID_SYSTEMS = frozenset({"UMLS", "SNOMED"})

Runner = Callable[..., str]


class MedCATLicenseNotAcknowledgedError(RuntimeError):
    """Raised when the bridge is invoked without an explicit license acknowledgement."""


class MedCATBridgeError(RuntimeError):
    """Raised when the MedCAT subprocess fails or returns unparsable output."""


def license_acknowledged(env: Optional[Mapping[str, str]] = None) -> bool:
    """Return ``True`` if the caller has explicitly accepted the MedCAT license."""
    source = env if env is not None else os.environ
    value = str(source.get(LICENSE_ENV_VAR, "")).strip().lower()
    return value in {"1", "true", "yes"}


def _prompt_for_acknowledgement() -> bool:
    """Ask for interactive acceptance when a terminal is attached; else refuse."""
    if not sys.stdin.isatty():
        return False
    try:
        answer = input(
            "Accept the MedCAT Elastic License 2.0 terms to continue? [y/N]: "
        )
    except (EOFError, OSError):
        return False
    return answer.strip().lower() in {"y", "yes"}


def ensure_license_acknowledged(
    *,
    env: Optional[Mapping[str, str]] = None,
    allow_interactive_prompt: bool = True,
) -> None:
    """Fail closed unless the MedCAT Elastic-2.0 license has been acknowledged.

    Always prints :data:`LICENSE_NOTICE` first. Acknowledgement is accepted
    either from ``OPENMED_ACCEPT_MEDCAT_LICENSE`` (see
    :func:`license_acknowledged`) or, when *allow_interactive_prompt* is true
    and a terminal is attached, from an interactive prompt.

    Raises:
        MedCATLicenseNotAcknowledgedError: If neither path acknowledges the
            license.
    """
    print(LICENSE_NOTICE, file=sys.stderr)

    if license_acknowledged(env):
        return
    if allow_interactive_prompt and _prompt_for_acknowledgement():
        return

    raise MedCATLicenseNotAcknowledgedError(
        "MedCAT invocation blocked: after reading and accepting MedCAT's "
        f"Elastic License 2.0 terms, set {LICENSE_ENV_VAR}=1 and retry."
    )


def _default_runner(
    command: Sequence[str],
    *,
    text: str,
    timeout: Optional[float],
    cwd: Optional[str],
    extra_env: Optional[Mapping[str, str]],
) -> str:
    """Pipe *text* to the MedCAT process stdin and return its stdout."""
    run_env = dict(os.environ)
    if extra_env:
        run_env.update(extra_env)

    completed = subprocess.run(
        list(command),
        input=text,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=cwd,
        env=run_env,
        check=False,
    )
    if completed.returncode != 0:
        raise MedCATBridgeError(
            f"MedCAT process exited with status {completed.returncode}: "
            f"{completed.stderr.strip()[:500]}"
        )
    return completed.stdout


def run_medcat(
    text: str,
    *,
    command: Sequence[str],
    default_system: str = DEFAULT_SYSTEM,
    timeout: Optional[float] = 60.0,
    cwd: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
    allow_interactive_prompt: bool = True,
    runner: Optional[Runner] = None,
) -> list[dict[str, Any]]:
    """Run a user-provided MedCAT process over *text* and map its concepts.

    MedCAT is never imported in-process (see :data:`LICENSE_NOTICE`).
    *command* must point at a script/CLI the caller has installed and
    configured themselves; OpenMed only pipes *text* to its stdin and parses
    JSON concept records from its stdout.

    Args:
        text: Input text to send to the MedCAT process via stdin.
        command: Argv for the subprocess, e.g. ``["medcat-cli", "--config", ...]``.
        default_system: Vocabulary system for records that do not carry
            their own ``ontology``/``system`` field. Must be ``"UMLS"`` or
            ``"SNOMED"``.
        timeout: Seconds to wait for the subprocess before raising.
        cwd: Optional working directory for the subprocess.
        env: Optional extra environment variables merged into the subprocess
            environment, and consulted for the license acknowledgement.
        allow_interactive_prompt: Whether to fall back to an interactive
            terminal prompt when the license env var is unset.
        runner: Injectable subprocess runner, primarily for tests. Defaults
            to a real ``subprocess.run`` invocation.

    Returns:
        Concept records mapped onto OpenMed span-code fields
        (``{"system", "code", "score"}``, plus ``"name"`` when present).

    Raises:
        MedCATLicenseNotAcknowledgedError: If the license has not been
            acknowledged and no interactive acceptance was given.
        MedCATBridgeError: If the subprocess fails or emits unparsable output.
    """
    if default_system not in _VALID_SYSTEMS:
        raise ValueError(f"default_system must be one of {sorted(_VALID_SYSTEMS)}")

    ensure_license_acknowledged(
        env=env, allow_interactive_prompt=allow_interactive_prompt
    )

    active_runner = runner or _default_runner
    raw_output = active_runner(
        command, text=text, timeout=timeout, cwd=cwd, extra_env=env
    )
    return parse_medcat_output(raw_output, default_system=default_system)


def parse_medcat_output(
    raw_output: str, *, default_system: str = DEFAULT_SYSTEM
) -> list[dict[str, Any]]:
    """Parse MedCAT subprocess stdout into OpenMed span codes.

    Accepts a JSON array of concept records, a top-level object with a
    ``"concepts"``/``"entities"`` list, or newline-delimited JSON (one
    record per line). Each record is shaped like
    ``{"cui": "C0011860", "name": "Diabetes Mellitus", "score": 0.98}``,
    with an optional ``"ontology"``/``"system"`` field overriding
    *default_system*.
    """
    if default_system not in _VALID_SYSTEMS:
        raise ValueError(f"default_system must be one of {sorted(_VALID_SYSTEMS)}")

    records = _parse_records(raw_output)
    return [
        _record_to_span_code(record, default_system=default_system)
        for record in records
    ]


def _parse_records(raw_output: str) -> list[Mapping[str, Any]]:
    stripped = raw_output.strip()
    if not stripped:
        return []

    try:
        parsed: Any = json.loads(stripped)
    except json.JSONDecodeError:
        records: list[Mapping[str, Any]] = []
        for line in stripped.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise MedCATBridgeError(
                    "Could not parse MedCAT output as JSON or newline-delimited JSON"
                ) from exc
        return records

    if isinstance(parsed, Mapping):
        parsed = parsed.get("concepts") or parsed.get("entities") or [parsed]
    if not isinstance(parsed, list):
        raise MedCATBridgeError("MedCAT output must be a JSON array of concept records")
    return parsed


def _record_to_span_code(
    record: Mapping[str, Any], *, default_system: str
) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise MedCATBridgeError(
            f"MedCAT concept record must be an object, got {type(record).__name__}"
        )

    cui = record.get("cui")
    if not cui:
        raise MedCATBridgeError("MedCAT concept record is missing required field 'cui'")

    system = str(
        record.get("ontology") or record.get("system") or default_system
    ).upper()
    if system not in _VALID_SYSTEMS:
        system = default_system

    payload: dict[str, Any] = {"system": system, "code": str(cui)}

    score = record.get("score")
    if score is not None:
        payload["score"] = float(score)

    name = record.get("name")
    if name is not None:
        payload["name"] = name

    return payload


__all__ = [
    "LICENSE_ENV_VAR",
    "LICENSE_NOTICE",
    "DEFAULT_SYSTEM",
    "MedCATLicenseNotAcknowledgedError",
    "MedCATBridgeError",
    "license_acknowledged",
    "ensure_license_acknowledged",
    "run_medcat",
    "parse_medcat_output",
]
