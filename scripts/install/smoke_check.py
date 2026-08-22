#!/usr/bin/env python3
"""Run a deterministic, offline smoke check against an installed OpenMed."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

SCHEMA_VERSION = 1
COMMAND_TIMEOUT_SECONDS = 30

_OFFLINE_FLAGS = {
    "OPENMED_OFFLINE": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
}
_MANIFEST_ROW_RE = re.compile(r"\(([1-9][0-9]{0,8}) rows? checked\)")
_MAX_VERSION_LENGTH = 64
_SAFE_VERSION_PATTERN = (
    r"[0-9]+(?:\.[0-9]+){2}"
    r"(?:(?:a|b|rc)[0-9]+)?"
    r"(?:\.post[0-9]+)?"
    r"(?:\.dev[0-9]+)?"
    r"(?:\+[0-9A-Za-z]+(?:[._-][0-9A-Za-z]+)*)?"
)
_VERSION_RE = re.compile(rf"^openmed ({_SAFE_VERSION_PATTERN})$")
_SAFE_VERSION_RE = re.compile(rf"^{_SAFE_VERSION_PATTERN}$")
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

# The probe carries only a synthetic marker. It emits hashes and offsets, never
# the marker itself, so captured child output remains safe for release logs.
_SYNTHETIC_PROBE = r"""
import json
from importlib.metadata import distribution
from types import SimpleNamespace

from openmed import redaction_preview

text = "Synthetic install marker OM-SMOKE-0001"
marker = "OM-SMOKE-0001"
start = text.index(marker)
result = SimpleNamespace(
    deidentified_text=text.replace(marker, "[ID_NUM]"),
    method="mask",
    pii_entities=[
        SimpleNamespace(
            action="mask",
            end=start + len(marker),
            label="ID_NUM",
            redacted_text="[ID_NUM]",
            start=start,
        )
    ],
)
preview = redaction_preview(text, result)
installed_distribution = distribution("openmed")
entry_point_declared = any(
    item.name == "openmed" and item.value == "openmed.cli:main"
    for item in installed_distribution.entry_points
)
if not entry_point_declared or preview["change_count"] != 1:
    raise SystemExit(1)
change = preview["changes"][0]
print(
    json.dumps(
        {
            "change_count": preview["change_count"],
            "document_hash": preview["document_hash"],
            "entry_point_declared": entry_point_declared,
            "package_version": installed_distribution.version,
            "surface_hash": change["surface_hash"],
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
)
"""


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class CheckResult:
    """One privacy-safe smoke-check result."""

    name: str
    status: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON representation for this check."""
        return {"name": self.name, "status": self.status, **dict(self.details)}


@dataclass(frozen=True)
class SmokeReport:
    """Compact, machine-readable evidence for one install smoke check."""

    status: str
    checks: tuple[CheckResult, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a stable, JSON-serializable report."""
        return {
            "checks": [check.to_dict() for check in self.checks],
            "offline": True,
            "schema_version": SCHEMA_VERSION,
            "status": self.status,
        }

    def to_json(self) -> str:
        """Render the report as one compact JSON document."""
        return json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )


def _passed(name: str, **details: Any) -> CheckResult:
    return CheckResult(name=name, status="passed", details=details)


def _failed(name: str, reason: str) -> CheckResult:
    return CheckResult(name=name, status="failed", details={"reason": reason})


def _clean_environment(home: Path) -> dict[str, str]:
    """Build a minimal child environment with network and user state disabled."""
    temporary = home / "tmp"
    cache = home / "cache"
    config = home / "config"
    temporary.mkdir()
    cache.mkdir()
    config.mkdir()

    environment = {
        "HOME": str(home),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": os.environ.get("PATH", os.defpath),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TMPDIR": str(temporary),
        "TEMP": str(temporary),
        "TMP": str(temporary),
        "XDG_CACHE_HOME": str(cache),
        "XDG_CONFIG_HOME": str(config),
    }
    environment.update(_OFFLINE_FLAGS)
    if os.name == "nt":  # pragma: no cover - Windows-only environment names.
        environment["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", "")
        environment["USERPROFILE"] = str(home)
    return environment


def _python_path(
    python_executable: str,
    *,
    path: str,
) -> Path | None:
    """Resolve the selected interpreter without dereferencing venv symlinks."""
    discovered = shutil.which(python_executable, path=path)
    if discovered is None:
        return None
    candidate = Path(discovered)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate


def _entry_point_path(python_path: Path) -> Path | None:
    """Find the console script installed beside the selected interpreter."""
    script_name = "openmed.exe" if os.name == "nt" else "openmed"
    adjacent = python_path.parent / script_name
    if adjacent.is_file():
        return adjacent
    return None


def _run_captured(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    runner: CommandRunner,
) -> subprocess.CompletedProcess[str] | None:
    """Run a child command without exposing its stdout or stderr."""
    try:
        return runner(
            list(command),
            capture_output=True,
            check=False,
            cwd=cwd,
            env=dict(environment),
            text=True,
            timeout=COMMAND_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def _parse_version(output: str | None) -> str | None:
    if not output or len(output) > len("openmed ") + _MAX_VERSION_LENGTH + 1:
        return None
    match = _VERSION_RE.fullmatch(output.strip())
    return match.group(1) if match else None


def _manifest_check(
    result: subprocess.CompletedProcess[str] | None,
) -> CheckResult:
    if result is None or result.returncode != 0:
        return _failed("bundled_manifest", "command_failed")
    try:
        payload = json.loads(result.stdout or "")
    except (TypeError, json.JSONDecodeError):
        return _failed("bundled_manifest", "invalid_json")

    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        return _failed("bundled_manifest", "invalid_result")
    if not isinstance(data, dict) or data.get("ok") is not True:
        return _failed("bundled_manifest", "asset_validation_failed")
    if data.get("violation_count") != 0:
        return _failed("bundled_manifest", "asset_validation_failed")

    row_count = None
    messages = data.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, str):
                continue
            match = _MANIFEST_ROW_RE.search(message)
            if match:
                row_count = int(match.group(1))
                break

    if row_count is None:
        return _failed("bundled_manifest", "invalid_result")
    return _passed("bundled_manifest", manifest_rows=row_count)


def _synthetic_check(
    python_executable: str,
    *,
    expected_version: str,
    cwd: Path,
    environment: Mapping[str, str],
    runner: CommandRunner,
) -> CheckResult:
    command = [python_executable, "-I", "-c", _SYNTHETIC_PROBE]
    first = _run_captured(
        command,
        cwd=cwd,
        environment=environment,
        runner=runner,
    )
    second = _run_captured(
        command,
        cwd=cwd,
        environment=environment,
        runner=runner,
    )
    if (
        first is None
        or second is None
        or first.returncode != 0
        or second.returncode != 0
    ):
        return _failed("synthetic_offline_command", "command_failed")
    if first.stdout != second.stdout:
        return _failed("synthetic_offline_command", "non_deterministic_output")

    try:
        payload = json.loads(first.stdout or "")
    except (TypeError, json.JSONDecodeError):
        return _failed("synthetic_offline_command", "invalid_json")
    if not isinstance(payload, dict):
        return _failed("synthetic_offline_command", "invalid_result")
    if payload.get("entry_point_declared") is not True:
        return _failed("synthetic_offline_command", "entry_point_not_declared")
    package_version = payload.get("package_version")
    if (
        not isinstance(package_version, str)
        or len(package_version) > _MAX_VERSION_LENGTH
        or _SAFE_VERSION_RE.fullmatch(package_version) is None
    ):
        return _failed("synthetic_offline_command", "invalid_package_version")
    if package_version != expected_version:
        return _failed("synthetic_offline_command", "version_mismatch")
    if payload.get("change_count") != 1:
        return _failed("synthetic_offline_command", "unexpected_change_count")
    if not all(
        isinstance(payload.get(key), str) and _HASH_RE.fullmatch(payload[key])
        for key in ("document_hash", "surface_hash")
    ):
        return _failed("synthetic_offline_command", "unsafe_or_invalid_hash")

    return _passed(
        "synthetic_offline_command",
        change_count=1,
        document_hash=payload["document_hash"],
        surface_hash=payload["surface_hash"],
    )


def run_smoke_check(
    *,
    python_executable: str = sys.executable,
    runner: CommandRunner = subprocess.run,
) -> SmokeReport:
    """Run the installed entry point and deterministic offline runtime probe.

    The selected interpreter must already contain an OpenMed installation. The
    checker never invokes a package installer or a network client; its clean
    temporary home only prevents host configuration, caches, and credentials
    from affecting the evidence.
    """
    with tempfile.TemporaryDirectory(prefix="openmed-install-smoke-") as raw_home:
        home = Path(raw_home)
        environment = _clean_environment(home)
        workdir = home / "work"
        workdir.mkdir()
        python_path = _python_path(
            python_executable,
            path=environment["PATH"],
        )
        if python_path is None:
            return SmokeReport(
                status="failed",
                checks=(_failed("entry_point", "python_not_found"),),
            )
        # Resolve a bare interpreter name using the caller's PATH, then remove
        # every unrelated environment from the child executable search path.
        environment["PATH"] = str(python_path.parent)
        entry_point = _entry_point_path(python_path)
        if entry_point is None:
            return SmokeReport(
                status="failed",
                checks=(_failed("entry_point", "not_installed"),),
            )

        version_result = _run_captured(
            [str(entry_point), "--version"],
            cwd=workdir,
            environment=environment,
            runner=runner,
        )
        version = _parse_version(
            version_result.stdout if version_result is not None else None
        )
        if version_result is None or version_result.returncode != 0:
            return SmokeReport(
                status="failed",
                checks=(_failed("entry_point", "command_failed"),),
            )
        if version is None:
            return SmokeReport(
                status="failed",
                checks=(_failed("entry_point", "invalid_version_output"),),
            )

        # Do not copy even a version-shaped child value into the report until
        # the isolated metadata probe confirms that it belongs to this install.
        checks = [_passed("entry_point")]
        manifest_result = _run_captured(
            [str(entry_point), "models", "validate", "--json"],
            cwd=workdir,
            environment=environment,
            runner=runner,
        )
        manifest_check = _manifest_check(manifest_result)
        checks.append(manifest_check)
        if manifest_check.status != "passed":
            return SmokeReport(status="failed", checks=tuple(checks))

        synthetic_check = _synthetic_check(
            str(python_path),
            expected_version=version,
            cwd=workdir,
            environment=environment,
            runner=runner,
        )
        checks.append(synthetic_check)
        status = "passed" if synthetic_check.status == "passed" else "failed"
        if status == "passed":
            checks[0] = _passed("entry_point", version=version)
        return SmokeReport(status=status, checks=tuple(checks))


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--python",
        dest="python_executable",
        default=sys.executable,
        help="Python executable for the installed OpenMed environment.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Print one smoke report and return its release-gate status."""
    args = _parse_args(argv)
    try:
        report = run_smoke_check(python_executable=args.python_executable)
    except Exception:  # pragma: no cover - defensive privacy boundary.
        report = SmokeReport(
            status="failed",
            checks=(_failed("smoke_check", "internal_error"),),
        )
    print(report.to_json())
    return 0 if report.status == "passed" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
