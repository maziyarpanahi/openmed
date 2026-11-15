"""Synthetic air-gap lifecycle kits and recovery hooks for rehearsal tests.

Everything here is synthetic and offline: the "releases" are tiny pure-Python
wheels named ``openmed`` whose only module records its version, and the model
cache holds a placeholder snapshot. No real checkpoint, credential, or patient
data is involved.
"""

from __future__ import annotations

import base64
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any
from unittest import mock

from openmed.__about__ import __version__
from openmed.cli import airgap
from openmed.cli.airgap_lifecycle import LifecycleInputs
from openmed.cli.airgap_rehearsal import HookContext, HookResult, SmokeResult

RELEASE_VERSION = __version__
UPGRADE_VERSION = f"{__version__}.post1"
_ZIP_TIME = (2026, 1, 1, 0, 0, 0)


def write_wheel(directory: Path, version: str, *, payload: str = "") -> Path:
    """Write a deterministic synthetic ``openmed`` wheel and return its path."""

    directory.mkdir(parents=True, exist_ok=True)
    dist_info = f"openmed-{version}.dist-info"
    files = {
        "openmed/__init__.py": (
            '"""Synthetic OpenMed release for air-gap rehearsal tests."""\n'
            f'__version__ = "{version}"\n{payload}'
        ),
        f"{dist_info}/METADATA": (
            f"Metadata-Version: 2.1\nName: openmed\nVersion: {version}\n"
        ),
        f"{dist_info}/WHEEL": (
            "Wheel-Version: 1.0\nGenerator: openmed-tests\n"
            "Root-Is-Purelib: true\nTag: py3-none-any\n"
        ),
    }
    rows = []
    for name, text in files.items():
        data = text.encode("utf-8")
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest())
        rows.append(f"{name},sha256={digest.rstrip(b'=').decode()},{len(data)}")
    rows.append(f"{dist_info}/RECORD,,")
    files[f"{dist_info}/RECORD"] = "\n".join(rows) + "\n"
    wheel = directory / f"openmed-{version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, text in files.items():
            archive.writestr(zipfile.ZipInfo(name, _ZIP_TIME), text)
    return wheel


def _fake_prefetch(model: str, *, cache_dir: str) -> str:
    repo_id = airgap.resolve_repo_id(model)
    snapshot = (
        Path(cache_dir)
        / f"models--{repo_id.replace('/', '--')}"
        / "snapshots"
        / "synthetic"
    )
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text('{"model_type":"synthetic"}')
    return str(snapshot)


def build_lifecycle_bundle(
    output: Path,
    workdir: Path,
    *,
    upgrade_version: str = UPGRADE_VERSION,
    rollback_version: str = RELEASE_VERSION,
    migrations: bool = True,
    recovery: bool = True,
    **lifecycle: Any,
) -> Path:
    """Build a real lifecycle bundle with synthetic wheels and model cache.

    Args:
        output: New bundle directory or ``.tar.gz`` path.
        workdir: Scratch directory for the caller-supplied input directories.
        upgrade_version: Version of the synthetic upgrade wheel.
        rollback_version: Version of the synthetic rollback wheel.
        migrations: Include one ordered migration asset.
        recovery: Include a synthetic recovery asset.
        **lifecycle: Extra :class:`LifecycleInputs` fields.

    Returns:
        The bundle path.
    """

    upgrade_dir = workdir / "upgrade"
    rollback_dir = workdir / "rollback"
    write_wheel(upgrade_dir, upgrade_version)
    write_wheel(rollback_dir, rollback_version)
    migrations_dir = recovery_dir = None
    if migrations:
        migrations_dir = workdir / "migrations"
        migrations_dir.mkdir(parents=True, exist_ok=True)
        (migrations_dir / "0001-synthetic-schema.json").write_text(
            '{"synthetic": true, "operation": "add_column"}\n', encoding="utf-8"
        )
    if recovery:
        recovery_dir = workdir / "recovery"
        (recovery_dir / "tools").mkdir(parents=True, exist_ok=True)
        (recovery_dir / "tools" / "restore-plan.json").write_text(
            '{"synthetic": true, "steps": ["verify", "restore"]}\n',
            encoding="utf-8",
        )

    def _fake_download(wheels_dir: Path, **_kwargs: Any) -> None:
        write_wheel(wheels_dir, RELEASE_VERSION)

    inputs = LifecycleInputs(
        upgrade=upgrade_dir,
        rollback=rollback_dir,
        migrations=migrations_dir,
        recovery=recovery_dir,
        **lifecycle,
    )
    with (
        mock.patch.object(airgap, "_download_wheels", _fake_download),
        mock.patch.object(airgap, "prefetch_model", _fake_prefetch),
    ):
        airgap.build_bundle(
            output,
            archive=str(output).endswith(".tar.gz"),
            lifecycle=inputs,
        )
    return output


_SYNTHETIC_DIGEST = "sha256:" + hashlib.sha256(b"synthetic-recovery").hexdigest()


def passing_hook(context: HookContext) -> HookResult:
    """A synthetic recovery hook that verifies its inputs and passes."""

    if context.hook == "migration" and not context.migrations:
        return HookResult("failed", "verification_failed")
    if context.hook == "backup_restore" and context.recovery_dir is None:
        return HookResult("failed", "verification_failed")
    if context.active_python is None or context.active_version is None:
        return HookResult("failed", "verification_failed")
    marker = context.state_dir / "verified"
    marker.write_text(context.hook, encoding="utf-8")
    return HookResult("passed", "ok", _SYNTHETIC_DIGEST)


def failing_hook(context: HookContext) -> HookResult:
    """A synthetic recovery hook that reports an audit-continuity gap."""

    return HookResult("failed", "continuity_gap")


def leaky_hook(context: HookContext) -> dict[str, Any]:
    """A hook that tries to put a value into evidence; it must be rejected."""

    return {"status": "failed", "reason_code": f"patient {context.state_dir}"}


def raising_hook(context: HookContext) -> HookResult:
    """A hook whose exception text must never reach the report."""

    raise RuntimeError(f"secret detail {context.state_dir}")


ALL_HOOKS = {
    name: "tests.fixtures.airgap_lifecycle:passing_hook"
    for name in (
        "migration",
        "backup_restore",
        "credential_reload",
        "model_activation",
        "audit_continuity",
        "durable_work_resume",
    )
}


class FakeInstaller:
    """Fast in-process stand-in for ``PipVenvInstaller``.

    A "slot" records the installed version and artifact digests; a disk fault
    is a marker file that the smoke check detects.
    """

    def __init__(self, *, smoke_versions: dict[str, str] | None = None) -> None:
        self.installs: list[str] = []
        self._smoke_versions = smoke_versions or {}

    def python(self, slot: Path) -> Path:
        return slot / "bin" / "python"

    def install(self, slot: Path, artifacts: Any) -> None:
        self.installs.append(slot.name)
        (slot / "bin").mkdir(parents=True)
        self.python(slot).write_text("", encoding="utf-8")
        version = artifacts[0][0].name.split("-")[1]
        record = {"version": version, "digests": [digest for _, digest in artifacts]}
        (slot / "release.json").write_text(
            json.dumps(record, sort_keys=True), encoding="utf-8"
        )

    def smoke(
        self, slot: Path, *, distribution: str, import_name: str, version: str
    ) -> SmokeResult:
        data = slot / "release.json"
        if not data.is_file():
            return SmokeResult(
                "prior_check_failed", "installed_files_missing", "prior_check_failed"
            )
        digest = "sha256:" + hashlib.sha256(data.read_bytes()).hexdigest()
        if (slot / "disk-fault").exists():
            return SmokeResult("prior_check_failed", "installed_files_modified", "ok")
        installed = (
            self._smoke_versions.get(slot.name)
            or json.loads(data.read_text(encoding="utf-8"))["version"]
        )
        return SmokeResult(
            "ok" if installed == version else "version_mismatch", "ok", "ok", digest
        )

    def inject_disk_fault(self, slot: Path, *, import_name: str) -> None:
        (slot / "disk-fault").write_text("synthetic", encoding="utf-8")
