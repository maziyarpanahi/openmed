"""Rehearse an air-gapped install, upgrade, rollback, and recovery offline.

``openmed airgap rehearse <bundle>`` proves that a lifecycle kit built by
``openmed airgap bundle`` (see :mod:`openmed.cli.airgap_lifecycle`) can move a
fresh installation through its whole lifecycle without reaching any external
service. Every phase runs inside the OS-enforced no-egress boundary of
``scripts/security/no_egress_gate.py``: the rehearsal reuses that gate's
boundary selection (Linux user + network namespace, or a container with
``--network none``), its hardened command wrapper, and its negative and
positive network controls, which run before and after the phases.

Phases run in a fixed order through a small state machine:

``preflight``
    Verify every artifact digest and every lifecycle decision (set membership,
    version order, compatibility window, rollback target, migration chain,
    target interpreter, validity window) before anything is installed.
``install``
    Clean install of the release set into a fresh virtual environment with
    ``pip install --no-index --no-deps --require-hashes``.
``smoke``
    Import the active release, verify its installed files against ``RECORD``,
    and run ``pip check``.
``upgrade``
    Install the upgrade set. The first attempt is interrupted at a declared
    fault point; a simulated restart must recover to exactly one declared state,
    and the resumed upgrade must commit exactly once.
``injected_failure``
    Corrupt an installed file of the upgraded release (a disk fault); the smoke
    check must detect it and admission must stay blocked.
``rollback``
    Reinstall the previous release from the rollback assets and verify the
    installation now matches the declared rollback target.
``restore``
    Pluggable backup-restore and credential-reload hooks.
``continuity``
    Verify the transaction ledger, then pluggable model-activation,
    audit-continuity, and durable-work hooks, and decide whether admission may
    resume.

Each transition is transactional: a journal records the intent, the active
pointer moves only after the staged release passed its smoke check, and the
last-green checkpoint is written last. Recovery always lands on the checkpoint.

Recovery features that are not present in this build report ``not_available``.
That is never a pass: the rehearsal is ``incomplete`` (exit 4) until every
recovery hook passes.

The rehearsal report is bounded and value-free: phase and check outcomes,
controlled reason codes, versions, counts, and digests only. It shares the
attestation envelope (``repro_hash`` and ``integrity_hash``) and is HMAC-signed
when the caller supplies a key, so it can be verified offline with
``openmed airgap verify-report``.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import hmac
import importlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from types import MappingProxyType, ModuleType
from typing import Any, Protocol

from ..core.attestation import (
    _BLOCKED_REASONS,
    _NEGATIVE_STATUSES,
    _POSITIVE_STATUSES,
    NO_EGRESS_BOUNDARY_MECHANISMS,
    NO_EGRESS_NEGATIVE_CONTROLS,
    NO_EGRESS_POSITIVE_CONTROLS,
    NO_EGRESS_RUNNER_FAILURE_REASONS,
    NO_EGRESS_SKIP_REASONS,
    AttestationReport,
    NoEgressAttestationError,
    _count_mapping,
    _iso_timestamp,
    _probe_records,
    _safe_commit,
    _safe_version,
)
from ..core.audit import stable_hash
from .airgap import (
    BUNDLE_FINDING_CODES,
    MANIFEST_NAME,
    bundle_findings,
    open_bundle_root,
)
from .airgap_lifecycle import (
    LIFECYCLE_REASON_CODES,
    RECOVERY_ROOT,
    SET_ROOTS,
    ArtifactSet,
    LifecyclePlan,
    set_digest,
    validate_lifecycle,
)

REHEARSAL_SCHEMA_VERSION = 1
REHEARSAL_REPORT_TYPE = "openmed.airgap_rehearsal"
REHEARSAL_VOLATILE_FIELDS = ("generated_at", "integrity_hash", "signature")
SIGNATURE_ALGORITHM = "HMAC-SHA256"
MAX_REPORT_BYTES = 65536
MIN_KEY_BYTES = 16
DEFAULT_TIMEOUT_SECONDS = 3600.0
DEFAULT_CONTAINER_TMPFS = "1g"

EXIT_PASSED = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_SKIPPED = 3
EXIT_INCOMPLETE = 4
STATUS_EXIT_CODES: Mapping[str, int] = MappingProxyType(
    {
        "passed": EXIT_PASSED,
        "failed": EXIT_FAILED,
        "skipped": EXIT_SKIPPED,
        "incomplete": EXIT_INCOMPLETE,
    }
)
"""Report status to process exit code. Only ``passed`` exits zero."""

PHASES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "preflight": (
            "bundle_integrity",
            "lifecycle_contract",
            "compatibility",
            "validity_window",
        ),
        "install": ("set_verified", "install_release", "activate_release"),
        "smoke": ("import_release", "installed_integrity", "dependency_check"),
        "upgrade": (
            "set_verified",
            "migration",
            "interrupted_attempt",
            "resume_upgrade",
            "smoke_upgraded",
        ),
        "injected_failure": (
            "disk_fault_injected",
            "fault_detected",
            "admission_blocked",
        ),
        "rollback": (
            "set_verified",
            "rollback_release",
            "smoke_rolled_back",
            "declared_state",
        ),
        "restore": ("backup_restore", "credential_reload"),
        "continuity": (
            "transaction_ledger",
            "model_activation",
            "audit_continuity",
            "durable_work_resume",
            "admission",
        ),
    }
)
"""Phase identifier to its ordered check identifiers (the state machine)."""

RECOVERY_HOOKS: Mapping[str, str] = MappingProxyType(
    {
        "migration": "upgrade",
        "backup_restore": "restore",
        "credential_reload": "restore",
        "model_activation": "continuity",
        "audit_continuity": "continuity",
        "durable_work_resume": "continuity",
    }
)
"""Pluggable recovery hook to the phase that calls it."""

BUILTIN_HOOKS: Mapping[str, str] = MappingProxyType({})
"""Recovery features shipped in this build, as ``hook -> "module:callable"``.

A feature that lands in-tree (backup/restore, migration rehearsal, model
activation, audit continuity) registers its rehearsal hook here; until then the
hook reports ``not_available``. ``--hook`` overrides an entry for one run.
"""

FAULT_POINTS = ("before_activate", "after_activate", "after_checkpoint")
"""Where the first upgrade attempt is interrupted."""
DEFAULT_FAULT_POINT = "after_activate"
RECOVERY_OUTCOMES = ("clean", "rolled_back_to_checkpoint", "resumed_committed")
TRANSACTIONS = ("install", "upgrade", "rollback")

CHECK_STATUSES = ("passed", "failed", "not_available", "not_run")
PHASE_STATUSES = ("passed", "failed", "incomplete", "not_run")
REPORT_STATUSES = ("passed", "failed", "incomplete", "skipped")
HOOK_PROVIDERS = ("builtin", "caller", "none")

HOOK_REASON_CODES = frozenset(
    {
        "ok",
        "not_required",
        "feature_not_present",
        "verification_failed",
        "integrity_mismatch",
        "schema_incompatible",
        "encryption_check_failed",
        "continuity_gap",
        "activation_failed",
        "credential_reload_failed",
        "resume_failed",
        "duplicate_work",
    }
)
"""Reason codes a recovery hook may return; anything else is rejected."""

CHECK_REASON_CODES = (
    frozenset(BUNDLE_FINDING_CODES)
    | LIFECYCLE_REASON_CODES
    | HOOK_REASON_CODES
    | {
        "bundle_unreadable",
        "installer_unavailable",
        "install_failed",
        "install_timeout",
        "checkpoint_failed",
        "distribution_missing",
        "version_mismatch",
        "import_failed",
        "installed_files_modified",
        "installed_files_missing",
        "dependency_conflict",
        "smoke_failed",
        "rolled_back_to_checkpoint",
        "resumed_committed",
        "fault_not_triggered",
        "state_ambiguous",
        "duplicate_commit",
        "journal_open",
        "declared_state_mismatch",
        "fault_detected",
        "fault_not_detected",
        "fault_injection_failed",
        "admission_blocked",
        "admission_not_blocked",
        "admission_resumed",
        "recovery_feature_unavailable",
        "recovery_check_failed",
        "hook_error",
        "hook_result_invalid",
        "prior_phase_failed",
        "prior_check_failed",
    }
)
"""Every controlled reason code a phase or check may carry."""

BOUNDARY_FAILURE_REASONS = frozenset(
    {
        "egress_not_blocked",
        "negative_control_incomplete",
        "boundary_not_isolated",
        "positive_control_failed",
    }
)
SKIP_REASONS = NO_EGRESS_SKIP_REASONS | {"gate_unavailable"}
REPORT_REASON_CODES = (
    CHECK_REASON_CODES
    | BOUNDARY_FAILURE_REASONS
    | SKIP_REASONS
    | NO_EGRESS_RUNNER_FAILURE_REASONS
    | {"rehearsal_passed"}
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
GATE_PATH = _REPO_ROOT / "scripts" / "security" / "no_egress_gate.py"
INNER_SCRIPT = _REPO_ROOT / "scripts" / "airgap" / "rehearse_inside.py"
CONTAINER_BUNDLE = "/airgap/bundle"
INNER_RESULT_KEYS = frozenset(
    {
        "kernel",
        "interfaces",
        "routes",
        "negative_controls",
        "positive_controls",
        "negative_controls_after",
        "rehearsal",
    }
)

_HOST_KINDS = frozenset({"linux", "darwin", "windows", "other"})
_REQUESTED_BOUNDARIES = ("auto", "netns", "container")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_VERSION_RE = re.compile(r"^[0-9][0-9A-Za-z.+-]{0,63}$")
_MIGRATION_ID_RE = re.compile(r"^\d{4}$")
_HOOK_SPEC_RE = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*"
    r":[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$"
)
_KEY_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_SUBPROCESS_TIMEOUT = 1800.0
_STATE_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class RehearsalError(ValueError):
    """A controlled rehearsal failure; the message is only the reason code."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class RehearsalReportError(RehearsalError):
    """Raised when a rehearsal report or in-boundary result is invalid."""


class InjectedFault(RuntimeError):
    """Simulated crash raised at a declared fault point."""

    def __init__(self, point: str) -> None:
        self.point = point
        super().__init__(point)


# ---------------------------------------------------------------------------
# Installer (venv + hash-locked pip, offline)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SmokeResult:
    """Value-free smoke outcome for one installed release.

    Attributes:
        import_release: ``ok`` or why the distribution failed to import.
        installed_integrity: ``ok`` or why installed files do not match RECORD.
        dependency_check: ``ok`` or ``dependency_conflict``.
        digest: Digest of the installed payload (paths and RECORD hashes).
    """

    import_release: str
    installed_integrity: str
    dependency_check: str
    digest: str | None = None

    @property
    def reason(self) -> str:
        """Return the first failing reason, or ``ok``."""

        for reason in (self.installed_integrity, self.import_release):
            if reason != "ok":
                return reason
        return self.dependency_check


class ReleaseInstaller(Protocol):
    """Installs one verified release set into an isolated slot."""

    def install(self, slot: Path, artifacts: Sequence[tuple[Path, str]]) -> None:
        """Install ``(path, sha256_hex)`` artifacts into a new slot."""

    def smoke(
        self, slot: Path, *, distribution: str, import_name: str, version: str
    ) -> SmokeResult:
        """Verify and import the release installed in ``slot``."""

    def inject_disk_fault(self, slot: Path, *, import_name: str) -> None:
        """Corrupt one installed file of ``import_name`` in ``slot``."""

    def python(self, slot: Path) -> Path:
        """Return the slot's interpreter."""


_SMOKE_PROGRAM = r"""
import base64, hashlib, importlib, json, sys
import importlib.metadata as metadata
distribution, import_name, expected = sys.argv[1:4]
out = {"import_release": "ok", "installed_integrity": "ok", "digest": None}
try:
    dist = metadata.distribution(distribution)
except metadata.PackageNotFoundError:
    out["import_release"] = "distribution_missing"
    out["installed_integrity"] = "installed_files_missing"
    print(json.dumps(out))
    raise SystemExit(0)
if (dist.version or "").lower() != expected.lower():
    out["import_release"] = "version_mismatch"
rows = []
for item in dist.files or ():
    if item.hash is None:
        continue
    name = str(item).replace("\\", "/")
    try:
        with open(item.locate(), "rb") as handle:
            data = handle.read()
    except OSError:
        out["installed_integrity"] = "installed_files_missing"
        continue
    actual = hashlib.new(item.hash.mode, data).digest()
    encoded = base64.urlsafe_b64encode(actual).rstrip(b"=").decode("ascii")
    if encoded != item.hash.value and out["installed_integrity"] == "ok":
        out["installed_integrity"] = "installed_files_modified"
    if ".dist-info/" not in name and not name.startswith("../"):
        rows.append([name, item.hash.mode, item.hash.value])
if not rows and out["installed_integrity"] == "ok":
    out["installed_integrity"] = "installed_files_missing"
if out["installed_integrity"] != "ok":
    out["import_release"] = "prior_check_failed"
elif out["import_release"] == "ok":
    try:
        importlib.import_module(import_name)
    except Exception:
        out["import_release"] = "import_failed"
rows.sort()
payload = json.dumps(rows, separators=(",", ":")).encode("utf-8")
out["digest"] = "sha256:" + hashlib.sha256(payload).hexdigest()
print(json.dumps(out))
"""

_LOCATE_PROGRAM = (
    "import importlib.util, sys\n"
    "spec = importlib.util.find_spec(sys.argv[1])\n"
    "print(spec.origin if spec and spec.origin else '')\n"
)


def _subprocess_env() -> dict[str, str]:
    keep = ("PATH", "SYSTEMROOT", "HOME", "TMPDIR", "TEMP", "TMP", "LANG", "LC_ALL")
    env = {key: os.environ[key] for key in keep if key in os.environ}
    env.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PIP_NO_INPUT": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "OPENMED_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    return env


class PipVenvInstaller:
    """Install each release into its own virtual environment, offline.

    ``python -m venv`` bootstraps pip from the interpreter's bundled wheel, and
    ``pip install --no-index --no-deps --require-hashes`` installs exactly the
    verified files: pip re-checks every digest at install time, so a file that
    changes after preflight is refused rather than installed.
    """

    def __init__(
        self,
        *,
        python: str | None = None,
        verbose: bool = False,
        timeout: float = _SUBPROCESS_TIMEOUT,
    ) -> None:
        self._base_python = python or sys.executable
        self._verbose = verbose
        self._timeout = timeout

    def python(self, slot: Path) -> Path:
        """Return the slot's interpreter path."""

        if os.name == "nt":
            return slot / "Scripts" / "python.exe"
        return slot / "bin" / "python"

    def _run(self, command: Sequence[str], *, failure: str) -> int:
        try:
            completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
                list(command),
                env=_subprocess_env(),
                stdin=subprocess.DEVNULL,
                stdout=2 if self._verbose else subprocess.DEVNULL,
                stderr=None if self._verbose else subprocess.DEVNULL,
                timeout=self._timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RehearsalError("install_timeout") from exc
        except OSError as exc:
            raise RehearsalError(failure) from exc
        return completed.returncode

    def install(self, slot: Path, artifacts: Sequence[tuple[Path, str]]) -> None:
        """Create a venv at ``slot`` and install the hash-locked artifacts."""

        slot.parent.mkdir(parents=True, exist_ok=True)
        if self._run(
            [self._base_python, "-m", "venv", str(slot)],
            failure="installer_unavailable",
        ):
            raise RehearsalError("installer_unavailable")
        interpreter = self.python(slot)
        if not interpreter.exists():
            raise RehearsalError("installer_unavailable")
        requirements = slot.parent / f".{slot.name}.requirements.txt"
        requirements.write_text(
            "".join(
                f"{path.resolve().as_uri()} --hash=sha256:{digest}\n"
                for path, digest in artifacts
            ),
            encoding="utf-8",
        )
        try:
            code = self._run(
                [
                    str(interpreter),
                    "-m",
                    "pip",
                    "--isolated",
                    "--disable-pip-version-check",
                    "--no-input",
                    "install",
                    "--no-index",
                    "--no-deps",
                    "--no-cache-dir",
                    "--no-compile",
                    "--no-warn-script-location",
                    "--require-hashes",
                    "--requirement",
                    str(requirements),
                ],
                failure="installer_unavailable",
            )
        finally:
            requirements.unlink(missing_ok=True)
        if code:
            raise RehearsalError("install_failed")

    def smoke(
        self, slot: Path, *, distribution: str, import_name: str, version: str
    ) -> SmokeResult:
        """Verify installed files against RECORD, import, and ``pip check``."""

        interpreter = str(self.python(slot))
        try:
            completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
                [
                    interpreter,
                    "-I",
                    "-B",
                    "-c",
                    _SMOKE_PROGRAM,
                    distribution,
                    import_name,
                    version,
                ],
                env=_subprocess_env(),
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return SmokeResult("smoke_failed", "smoke_failed", "prior_check_failed")
        lines = (completed.stdout or "").strip().splitlines()
        try:
            result = json.loads(lines[-1]) if lines else None
        except json.JSONDecodeError:
            result = None
        if completed.returncode != 0 or not isinstance(result, dict):
            return SmokeResult("smoke_failed", "smoke_failed", "prior_check_failed")
        dependency = "ok"
        code = self._run(
            [
                interpreter,
                "-m",
                "pip",
                "--isolated",
                "--disable-pip-version-check",
                "--no-input",
                "check",
            ],
            failure="smoke_failed",
        )
        if code:
            dependency = "dependency_conflict"
        digest = result.get("digest")
        return SmokeResult(
            import_release=_reason_or(result.get("import_release"), "smoke_failed"),
            installed_integrity=_reason_or(
                result.get("installed_integrity"), "smoke_failed"
            ),
            dependency_check=dependency,
            digest=digest
            if isinstance(digest, str) and _DIGEST_RE.match(digest)
            else None,
        )

    def inject_disk_fault(self, slot: Path, *, import_name: str) -> None:
        """Append a stray byte to the release's top-level module file."""

        try:
            completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
                [
                    str(self.python(slot)),
                    "-I",
                    "-B",
                    "-c",
                    _LOCATE_PROGRAM,
                    import_name,
                ],
                env=_subprocess_env(),
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise RehearsalError("fault_injection_failed") from exc
        origin = (completed.stdout or "").strip()
        target = Path(origin) if origin else None
        if completed.returncode != 0 or target is None or not target.is_file():
            raise RehearsalError("fault_injection_failed")
        try:
            target.resolve().relative_to(slot.resolve())
            with target.open("ab") as handle:
                handle.write(b"\x00")
        except (OSError, ValueError) as exc:
            raise RehearsalError("fault_injection_failed") from exc


def _reason_or(value: Any, fallback: str) -> str:
    return value if isinstance(value, str) and value in CHECK_REASON_CODES else fallback


# ---------------------------------------------------------------------------
# Transactional installation state machine
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReleaseRecord:
    """One declared installation state: which slot runs which release set."""

    slot: str
    set_id: str
    version: str
    set_digest: str

    def to_dict(self) -> dict[str, str]:
        """Return the JSON form stored in state files."""

        return {
            "slot": self.slot,
            "set_id": self.set_id,
            "version": self.version,
            "set_digest": self.set_digest,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> ReleaseRecord | None:
        """Parse a stored record; ``None`` stays ``None``."""

        if raw is None:
            return None
        if not isinstance(raw, Mapping) or set(raw) != {
            "slot",
            "set_id",
            "version",
            "set_digest",
        }:
            raise RehearsalError("state_ambiguous")
        slot, set_id = raw["slot"], raw["set_id"]
        version, digest = raw["version"], raw["set_digest"]
        if (
            slot not in TRANSACTIONS
            or set_id not in SET_ROOTS
            or not isinstance(version, str)
            or not _VERSION_RE.match(version)
            or not isinstance(digest, str)
            or not _DIGEST_RE.match(digest)
        ):
            raise RehearsalError("state_ambiguous")
        return cls(slot, set_id, version, digest)


@dataclass(frozen=True)
class StagedSet:
    """A lifecycle set whose files were re-verified immediately before use."""

    set_id: str
    version: str
    digest: str
    artifacts: tuple[tuple[Path, str], ...]


@dataclass(frozen=True)
class TransitionResult:
    """Outcome of one transition: ``committed`` or ``already_committed``."""

    outcome: str
    record: ReleaseRecord | None
    smoke: SmokeResult | None = None


@dataclass(frozen=True)
class RecoveryResult:
    """Outcome of :meth:`LifecycleInstallation.recover`."""

    outcome: str
    declared: ReleaseRecord | None


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON via a temporary file, ``fsync``, and an atomic replace."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    if os.name == "posix":
        try:
            descriptor = os.open(path.parent, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(descriptor)
        except OSError:
            pass
        finally:
            os.close(descriptor)


def _read_json(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise RehearsalError("state_ambiguous") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise RehearsalError("state_ambiguous") from exc


class LifecycleInstallation:
    """A transactional installation root with a last-green checkpoint.

    Layout::

        releases/<transaction>/   one isolated environment per transaction
        state/active.json         the active pointer (moved atomically)
        state/checkpoint.json     the last-green declared state and ledger
        state/journal.json        an in-flight transition, if any

    A transition writes its intent to the journal, builds and smoke-tests the
    new release in its own slot, moves the active pointer, writes the
    checkpoint (appending the transaction to the committed ledger), and only
    then clears the journal. :meth:`recover` resolves any open journal to
    exactly one declared state: the checkpoint.
    """

    def __init__(
        self,
        root: Path,
        installer: ReleaseInstaller,
        *,
        fault_point: str | None = None,
    ) -> None:
        if fault_point is not None and fault_point not in FAULT_POINTS:
            raise RehearsalError("fault_point_invalid")
        self.root = root
        self.installer = installer
        self._fault_point = fault_point
        self.releases = root / "releases"
        self.state_dir = root / "state"
        self.checkpoint_path = self.state_dir / "checkpoint.json"
        self.pointer_path = self.state_dir / "active.json"
        self.journal_path = self.state_dir / "journal.json"

    def slot_path(self, record: ReleaseRecord) -> Path:
        """Return the slot directory that holds ``record``'s release."""

        return self.releases / record.slot

    def checkpoint(self) -> dict[str, Any]:
        """Return the last-green checkpoint (an empty one before install)."""

        raw = _read_json(self.checkpoint_path)
        if raw is None:
            return {
                "schema_version": _STATE_SCHEMA_VERSION,
                "active": None,
                "committed": [],
                "last_green": None,
            }
        if (
            not isinstance(raw, dict)
            or raw.get("schema_version") != _STATE_SCHEMA_VERSION
            or not isinstance(raw.get("committed"), list)
            or not all(item in TRANSACTIONS for item in raw["committed"])
        ):
            raise RehearsalError("state_ambiguous")
        ReleaseRecord.from_dict(raw.get("active"))
        return raw

    def declared(self) -> ReleaseRecord | None:
        """Return the checkpoint's declared active release."""

        return ReleaseRecord.from_dict(self.checkpoint().get("active"))

    def active(self) -> ReleaseRecord | None:
        """Return the release the active pointer currently selects."""

        return ReleaseRecord.from_dict(_read_json(self.pointer_path))

    def journal(self) -> dict[str, Any] | None:
        """Return the open journal entry, if a transition is in flight."""

        raw = _read_json(self.journal_path)
        if raw is None:
            return None
        if (
            not isinstance(raw, dict)
            or raw.get("txn") not in TRANSACTIONS
            or raw.get("stage") not in {"preparing", "prepared", "activated"}
        ):
            raise RehearsalError("state_ambiguous")
        ReleaseRecord.from_dict(raw.get("target"))
        ReleaseRecord.from_dict(raw.get("prior"))
        return raw

    def _fault(self, point: str) -> None:
        if self._fault_point == point:
            self._fault_point = None
            raise InjectedFault(point)

    def transition(
        self,
        txn: str,
        staged: StagedSet,
        *,
        distribution: str,
        import_name: str,
    ) -> TransitionResult:
        """Move the installation to ``staged`` as transaction ``txn``.

        A transaction already in the committed ledger is not applied again,
        so a resumed transition never duplicates committed work.

        Args:
            txn: ``install``, ``upgrade``, or ``rollback``.
            staged: The verified set to install.
            distribution: Distribution the smoke check verifies.
            import_name: Module the smoke check imports.

        Returns:
            ``committed`` with the new record, or ``already_committed``.

        Raises:
            RehearsalError: With a controlled code when installation, smoke
                verification, or a state write fails; the prior state stays
                declared and active.
            InjectedFault: At the armed fault point.
        """

        if txn not in TRANSACTIONS:
            raise RehearsalError("state_ambiguous")
        checkpoint = self.checkpoint()
        if txn in checkpoint["committed"]:
            return TransitionResult("already_committed", self.declared())
        if self.journal() is not None:
            raise RehearsalError("journal_open")
        prior = self.declared()
        target = ReleaseRecord(txn, staged.set_id, staged.version, staged.digest)
        journal: dict[str, Any] = {
            "schema_version": _STATE_SCHEMA_VERSION,
            "txn": txn,
            "target": target.to_dict(),
            "prior": prior.to_dict() if prior else None,
            "stage": "preparing",
        }
        try:
            _write_json_atomic(self.journal_path, journal)
        except OSError as exc:
            raise RehearsalError("checkpoint_failed") from exc
        slot = self.slot_path(target)
        try:
            if slot.exists():
                shutil.rmtree(slot)
            self.installer.install(slot, staged.artifacts)
            smoke = self.installer.smoke(
                slot,
                distribution=distribution,
                import_name=import_name,
                version=staged.version,
            )
            if smoke.reason != "ok":
                raise RehearsalError(smoke.reason)
        except (RehearsalError, OSError) as exc:
            # Fail closed: drop the staged slot, keep the prior state active.
            shutil.rmtree(slot, ignore_errors=True)
            self.journal_path.unlink(missing_ok=True)
            if isinstance(exc, RehearsalError):
                raise
            raise RehearsalError("install_failed") from exc
        try:
            journal["stage"] = "prepared"
            _write_json_atomic(self.journal_path, journal)
            self._fault("before_activate")
            _write_json_atomic(self.pointer_path, target.to_dict())
            journal["stage"] = "activated"
            _write_json_atomic(self.journal_path, journal)
            self._fault("after_activate")
            checkpoint = dict(checkpoint)
            checkpoint.update(
                {
                    "active": target.to_dict(),
                    "committed": [*checkpoint["committed"], txn],
                    "last_green": txn,
                }
            )
            _write_json_atomic(self.checkpoint_path, checkpoint)
            self._fault("after_checkpoint")
            self.journal_path.unlink()
        except OSError as exc:
            raise RehearsalError("checkpoint_failed") from exc
        return TransitionResult("committed", target, smoke)

    def recover(self) -> RecoveryResult:
        """Resolve any interrupted transition to the checkpoint's state.

        An uncommitted transition is rolled back: its slot is removed and the
        active pointer returns to the declared release. A transition whose
        checkpoint was written is resumed: only its journal is cleared.

        Returns:
            ``clean``, ``rolled_back_to_checkpoint``, or ``resumed_committed``
            with the single declared state.

        Raises:
            RehearsalError: ``state_ambiguous`` for unreadable state, or
                ``duplicate_commit`` for a ledger that committed twice.
        """

        checkpoint = self.checkpoint()
        declared = ReleaseRecord.from_dict(checkpoint.get("active"))
        committed = list(checkpoint["committed"])
        if len(set(committed)) != len(committed):
            raise RehearsalError("duplicate_commit")
        journal = self.journal()
        outcome = "clean"
        if journal is not None:
            if journal["txn"] in committed:
                outcome = "resumed_committed"
            else:
                outcome = "rolled_back_to_checkpoint"
                target = ReleaseRecord.from_dict(journal["target"])
                if target is not None and (declared is None or target != declared):
                    shutil.rmtree(self.slot_path(target), ignore_errors=True)
        if self.active() != declared:
            if outcome == "clean":
                outcome = "rolled_back_to_checkpoint"
            if declared is None:
                self.pointer_path.unlink(missing_ok=True)
            else:
                _write_json_atomic(self.pointer_path, declared.to_dict())
        self.journal_path.unlink(missing_ok=True)
        return RecoveryResult(outcome, declared)


# ---------------------------------------------------------------------------
# Pluggable recovery hooks
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HookContext:
    """What a recovery hook may use. Paths never reach the report.

    Attributes:
        hook: The hook identifier (see :data:`RECOVERY_HOOKS`).
        phase: The phase calling the hook.
        active_version: Version of the active release.
        active_python: Interpreter of the active release slot.
        state_dir: Writable directory private to this hook.
        recovery_dir: Verified bundle recovery assets, if the kit has any.
        migrations: Verified migration assets for the upgrade edge, in order.
        from_version: Release version before the upgrade.
        to_version: Upgrade version.
    """

    hook: str
    phase: str
    active_version: str | None
    active_python: Path | None
    state_dir: Path
    recovery_dir: Path | None
    migrations: tuple[Path, ...]
    from_version: str | None
    to_version: str | None


@dataclass(frozen=True)
class HookResult:
    """Value-free result a recovery hook returns.

    Attributes:
        status: ``passed``, ``failed``, or ``not_available``.
        reason_code: One of :data:`HOOK_REASON_CODES`.
        digest: Optional ``sha256:`` digest of what the hook verified.
    """

    status: str
    reason_code: str = "ok"
    digest: str | None = None


RecoveryHook = Callable[[HookContext], "HookResult | Mapping[str, Any]"]


@dataclass(frozen=True)
class ResolvedHook:
    """A hook ready to call, with its provenance."""

    name: str
    provider: str
    function: RecoveryHook | None
    load_failed: bool = False


def parse_hook_specs(values: Sequence[str]) -> dict[str, str]:
    """Parse ``NAME=module:callable`` options.

    Args:
        values: Raw ``--hook`` values.

    Returns:
        Hook name to ``module:callable`` spec.

    Raises:
        RehearsalError: ``hook_spec_invalid`` or ``hook_unknown``.
    """

    specs: dict[str, str] = {}
    for value in values:
        name, separator, spec = value.partition("=")
        if not separator or not _HOOK_SPEC_RE.match(spec):
            raise RehearsalError("hook_spec_invalid")
        if name not in RECOVERY_HOOKS:
            raise RehearsalError("hook_unknown")
        if name in specs:
            raise RehearsalError("hook_spec_invalid")
        specs[name] = spec
    return specs


def _load_callable(spec: str) -> RecoveryHook:
    module_name, _, attribute = spec.partition(":")
    target: Any = importlib.import_module(module_name)
    for part in attribute.split("."):
        target = getattr(target, part)
    if not callable(target):
        raise TypeError("hook is not callable")
    return target


def resolve_hooks(
    caller_specs: Mapping[str, str] | None = None,
) -> dict[str, ResolvedHook]:
    """Resolve every recovery hook from caller specs, then built-in features.

    A built-in feature that cannot be imported is ``not_available``. A caller
    hook that cannot be imported resolves to a hook that fails with
    ``hook_error``: a requested hook never silently becomes optional.

    Args:
        caller_specs: Parsed ``--hook`` values.

    Returns:
        Hook name to its resolved callable (``None`` when not present).
    """

    resolved: dict[str, ResolvedHook] = {}
    for name in RECOVERY_HOOKS:
        spec = (caller_specs or {}).get(name)
        provider = "caller"
        if spec is None:
            spec = BUILTIN_HOOKS.get(name)
            provider = "builtin"
        if spec is None:
            resolved[name] = ResolvedHook(name, "none", None)
            continue
        try:
            function: RecoveryHook | None = _load_callable(spec)
        except Exception:  # an unimportable plug-in proves nothing
            if provider == "builtin":
                resolved[name] = ResolvedHook(name, "none", None)
                continue
            resolved[name] = ResolvedHook(name, provider, None, load_failed=True)
            continue
        resolved[name] = ResolvedHook(name, provider, function)
    return resolved


def run_hook(hook: ResolvedHook, context: HookContext) -> dict[str, Any]:
    """Call a hook and normalize its result to a value-free check record."""

    if hook.load_failed:
        return _check("failed", "hook_error", provider=hook.provider)
    if hook.function is None:
        return _check("not_available", "feature_not_present", provider="none")
    try:
        raw = hook.function(context)
    except Exception:  # the exception text may carry values; drop it
        return _check("failed", "hook_error", provider=hook.provider)
    status: Any
    reason: Any
    digest: Any
    if isinstance(raw, HookResult):
        status, reason, digest = raw.status, raw.reason_code, raw.digest
    elif isinstance(raw, Mapping):
        status = raw.get("status")
        reason = raw.get("reason_code", "ok")
        digest = raw.get("digest")
    else:
        return _check("failed", "hook_result_invalid", provider=hook.provider)
    valid = (
        status in {"passed", "failed", "not_available"}
        and reason in HOOK_REASON_CODES
        and (digest is None or (isinstance(digest, str) and _DIGEST_RE.match(digest)))
        and (status != "passed" or reason in {"ok", "not_required"})
        and (status != "not_available" or reason == "feature_not_present")
        and (
            status != "failed"
            or reason not in {"ok", "not_required", "feature_not_present"}
        )
    )
    if not valid:
        return _check("failed", "hook_result_invalid", provider=hook.provider)
    return _check(str(status), str(reason), digest=digest, provider=hook.provider)


def _check(
    status: str,
    reason: str,
    *,
    digest: str | None = None,
    provider: str | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "reason_code": reason,
        "digest": digest,
        "provider": provider,
    }


# ---------------------------------------------------------------------------
# Phase runner (executes inside the no-egress boundary)
# ---------------------------------------------------------------------------


class _CheckFailed(Exception):
    def __init__(self, reason: str, digest: str | None = None) -> None:
        self.reason = reason
        self.digest = digest
        super().__init__(reason)


class _Rehearsal:
    """Drive the phase state machine over one bundle root."""

    def __init__(
        self,
        *,
        hooks: Mapping[str, ResolvedHook],
        installer: ReleaseInstaller,
        fault_point: str,
        now: datetime,
    ) -> None:
        self.hooks = hooks
        self.installer = installer
        self.fault_point = fault_point
        self.now = now
        self.records: dict[str, dict[str, dict[str, Any]]] = {
            phase: {} for phase in PHASES
        }
        self.plan: LifecyclePlan | None = None
        self.root: Path | None = None
        self.work: Path | None = None
        self.installation: LifecycleInstallation | None = None
        self.bundle: dict[str, Any] = _empty_bundle_summary()
        self.recovery_outcome: str | None = None
        self.disk_fault: str | None = None

    # -- bookkeeping -------------------------------------------------------

    def _mark(self, phase: str, check: str, record: Mapping[str, Any]) -> bool:
        self.records[phase][check] = dict(record)
        return record["status"] != "failed"

    def _step(self, phase: str, check: str, operation: Callable[[], Any]) -> bool:
        try:
            outcome = operation()
        except _CheckFailed as exc:
            return self._mark(
                phase, check, _check("failed", exc.reason, digest=exc.digest)
            )
        except RehearsalError as exc:
            reason = exc.code if exc.code in CHECK_REASON_CODES else "state_ambiguous"
            return self._mark(phase, check, _check("failed", reason))
        if isinstance(outcome, Mapping):
            return self._mark(phase, check, outcome)
        reason, digest = outcome if isinstance(outcome, tuple) else ("ok", outcome)
        return self._mark(phase, check, _check("passed", reason, digest=digest))

    def _phase_result(self, phase: str) -> dict[str, Any]:
        checks = []
        for check in PHASES[phase]:
            record = self.records[phase].get(check) or _check(
                "not_run", "prior_check_failed"
            )
            checks.append({"id": check, **record})
        status, reason = _phase_status(checks)
        return {"id": phase, "status": status, "reason_code": reason, "checks": checks}

    # -- driver -------------------------------------------------------------

    def run(self, root: Path, work: Path) -> dict[str, Any]:
        self.root = root
        self.work = work
        for phase in PHASES:
            handler = getattr(self, f"_phase_{phase}")
            try:
                handler()
            except (RehearsalError, InjectedFault, OSError) as exc:
                code = getattr(exc, "code", "state_ambiguous")
                self._fail_next(
                    phase, code if code in CHECK_REASON_CODES else "state_ambiguous"
                )
            if self._phase_result(phase)["status"] == "failed":
                break
        return self.result()

    def _fail_next(self, phase: str, reason: str) -> None:
        pending = [check for check in PHASES[phase] if check not in self.records[phase]]
        check = pending[0] if pending else PHASES[phase][-1]
        self._mark(phase, check, _check("failed", reason))

    def result(self) -> dict[str, Any]:
        phases = []
        failed = False
        for phase in PHASES:
            if failed or not self.records[phase]:
                phases.append(_not_run_phase(phase))
                failed = True
                continue
            record = self._phase_result(phase)
            phases.append(record)
            failed = failed or record["status"] == "failed"
        return {
            "bundle": self.bundle,
            "fault": {
                "upgrade_interruption": self.fault_point,
                "recovery": self.recovery_outcome,
                "disk_fault": self.disk_fault,
            },
            "phases": phases,
            "final_state": self._final_state(),
        }

    def _final_state(self) -> dict[str, Any] | None:
        if self.installation is None:
            return None
        try:
            checkpoint = self.installation.checkpoint()
            declared = self.installation.declared()
            journal = self.installation.journal()
        except RehearsalError:
            return None
        if declared is None:
            return None
        committed = list(checkpoint["committed"])
        return {
            "version": declared.version,
            "set_id": declared.set_id,
            "set_digest": declared.set_digest,
            "committed_transactions": len(committed),
            "duplicate_commits": len(committed) - len(set(committed)),
            "journal_open": journal is not None,
        }

    # -- helpers ------------------------------------------------------------

    def _staged(self, set_id: str) -> StagedSet:
        assert self.plan is not None and self.root is not None
        artifact_set = self.plan.sets.get(set_id)
        if artifact_set is None or artifact_set.version is None:
            raise _CheckFailed("set_missing")
        return StagedSet(
            set_id=set_id,
            version=artifact_set.version,
            digest=artifact_set.digest,
            artifacts=self._reverify(artifact_set),
        )

    def _reverify(self, artifact_set: ArtifactSet) -> tuple[tuple[Path, str], ...]:
        """Re-hash a set's files right before they are used."""

        assert self.root is not None
        root = self.root.resolve()
        verified: list[tuple[Path, str]] = []
        for relative, expected in artifact_set.artifacts:
            path = root / PurePosixPath(relative)
            try:
                resolved = path.resolve(strict=True)
                resolved.relative_to(root)
                if path.is_symlink() or not resolved.is_file():
                    raise _CheckFailed("artifact_unsafe")
                actual = _file_sha256(resolved)
            except FileNotFoundError as exc:
                raise _CheckFailed("artifact_missing") from exc
            except (OSError, ValueError) as exc:
                raise _CheckFailed("artifact_unreadable") from exc
            if actual != expected:
                raise _CheckFailed("artifact_modified")
            verified.append((resolved, actual))
        if set_digest(artifact_set.root, artifact_set.artifacts) != artifact_set.digest:
            raise _CheckFailed("set_digest_mismatch")
        return tuple(verified)

    def _smoke_active(self) -> tuple[ReleaseRecord, SmokeResult]:
        assert self.installation is not None and self.plan is not None
        active = self.installation.active()
        if active is None or active != self.installation.declared():
            raise _CheckFailed("declared_state_mismatch")
        smoke = self.installer.smoke(
            self.installation.slot_path(active),
            distribution=self.plan.distribution,
            import_name=self.plan.import_name,
            version=active.version,
        )
        return active, smoke

    def _admission(self) -> tuple[bool, str]:
        """Admission is open only for one clean, verified declared state."""

        assert self.installation is not None
        try:
            if self.installation.journal() is not None:
                return False, "journal_open"
            _, smoke = self._smoke_active()
        except _CheckFailed as exc:
            return False, exc.reason
        except RehearsalError as exc:
            return False, exc.code
        if smoke.reason != "ok":
            return False, smoke.reason
        return True, "ok"

    def _hook_context(self, hook: str, phase: str) -> HookContext:
        assert self.installation is not None and self.plan is not None
        assert self.work is not None and self.root is not None
        active = self.installation.active()
        state_dir = self.work / "hooks" / hook
        state_dir.mkdir(parents=True, exist_ok=True)
        recovery = self.plan.sets.get("recovery")
        migrations = tuple(
            self.root / PurePosixPath(migration.path)
            for migration in self.plan.migrations
        )
        return HookContext(
            hook=hook,
            phase=phase,
            active_version=active.version if active else None,
            active_python=(
                self.installer.python(self.installation.slot_path(active))
                if active
                else None
            ),
            state_dir=state_dir,
            recovery_dir=self.root / RECOVERY_ROOT if recovery else None,
            migrations=migrations,
            from_version=self.plan.sets["release"].version,
            to_version=self.plan.sets["upgrade"].version,
        )

    def _hook(self, phase: str, hook: str) -> bool:
        record = run_hook(self.hooks[hook], self._hook_context(hook, phase))
        return self._mark(phase, hook, record)

    # -- phases ---------------------------------------------------------------

    def _phase_preflight(self) -> None:
        assert self.root is not None
        manifest, findings = bundle_findings(self.root)
        manifest_digest = _file_digest_or_none(self.root / MANIFEST_NAME)
        codes = {finding.code for finding in findings}
        integrity = next((code for code in BUNDLE_FINDING_CODES if code in codes), "ok")
        self._mark(
            "preflight",
            "bundle_integrity",
            _check(
                "passed" if integrity == "ok" else "failed",
                integrity,
                digest=manifest_digest,
            ),
        )
        self.bundle = _bundle_summary(manifest, manifest_digest, None)
        if manifest is None:
            return
        validation = validate_lifecycle(manifest, now=self.now)
        for check in validation.checks:
            self._mark(
                "preflight", check.check_id, _check(check.status, check.reason_code)
            )
        self.plan = validation.plan if validation.passed and integrity == "ok" else None
        self.bundle = _bundle_summary(manifest, manifest_digest, validation.plan)

    def _phase_install(self) -> None:
        assert self.work is not None and self.plan is not None
        self.installation = LifecycleInstallation(
            self.work / "installation", self.installer
        )
        staged: list[StagedSet] = []

        def _verify() -> str:
            staged.append(self._staged("release"))
            return staged[0].digest

        if not self._step("install", "set_verified", _verify):
            return

        def _install() -> str | None:
            assert self.installation is not None and self.plan is not None
            result = self.installation.transition(
                "install",
                staged[0],
                distribution=self.plan.distribution,
                import_name=self.plan.import_name,
            )
            return result.smoke.digest if result.smoke else None

        if not self._step("install", "install_release", _install):
            return
        self._step("install", "activate_release", self._verify_declared("release"))

    def _verify_declared(self, set_id: str) -> Callable[[], str]:
        def _verify() -> str:
            assert self.installation is not None and self.plan is not None
            if self.installation.recover().outcome != "clean":
                raise _CheckFailed("state_ambiguous")
            active = self.installation.active()
            expected = self.plan.sets[set_id]
            if (
                active is None
                or active != self.installation.declared()
                or active.set_id != set_id
                or active.version != expected.version
                or active.set_digest != expected.digest
            ):
                raise _CheckFailed("declared_state_mismatch")
            return active.set_digest

        return _verify

    def _phase_smoke(self) -> None:
        _, smoke = self._smoke_active_or_fail("smoke", "import_release")
        if smoke is None:
            return
        for check, reason in (
            ("import_release", smoke.import_release),
            ("installed_integrity", smoke.installed_integrity),
            ("dependency_check", smoke.dependency_check),
        ):
            status = "passed" if reason == "ok" else "failed"
            if reason == "prior_check_failed":
                status = "not_run"
            self._mark("smoke", check, _check(status, reason, digest=smoke.digest))

    def _smoke_active_or_fail(
        self, phase: str, check: str
    ) -> tuple[ReleaseRecord | None, SmokeResult | None]:
        try:
            return self._smoke_active()
        except _CheckFailed as exc:
            self._mark(phase, check, _check("failed", exc.reason))
            return None, None

    def _phase_upgrade(self) -> None:
        assert self.plan is not None and self.work is not None
        staged: list[StagedSet] = []

        def _verify() -> str:
            staged.append(self._staged("upgrade"))
            for migration in self.plan.migrations if self.plan else ():
                path = self.root / PurePosixPath(migration.path) if self.root else None
                if path is None or not path.is_file():
                    raise _CheckFailed("artifact_missing")
                if _file_sha256(path) != migration.sha256:
                    raise _CheckFailed("artifact_modified")
            return staged[0].digest

        if not self._step("upgrade", "set_verified", _verify):
            return
        migration_hook = self.hooks["migration"]
        if self.plan.migrations or migration_hook.provider != "none":
            if not self._hook("upgrade", "migration"):
                return
        else:
            self._mark("upgrade", "migration", _check("passed", "not_required"))
        root = self.work / "installation"
        release = self.plan.sets["release"]
        upgrade = self.plan.sets["upgrade"]

        def _interrupted() -> tuple[str, str]:
            assert self.plan is not None and release.version and upgrade.version
            armed = LifecycleInstallation(
                root, self.installer, fault_point=self.fault_point
            )
            try:
                armed.transition(
                    "upgrade",
                    staged[0],
                    distribution=self.plan.distribution,
                    import_name=self.plan.import_name,
                )
            except InjectedFault:
                pass
            else:
                raise _CheckFailed("fault_not_triggered")
            restarted = LifecycleInstallation(root, self.installer)
            recovery = restarted.recover()
            self.installation = restarted
            self.recovery_outcome = recovery.outcome
            if self.fault_point == "after_checkpoint":
                expected = ("resumed_committed", "upgrade", upgrade.digest)
            else:
                expected = ("rolled_back_to_checkpoint", "release", release.digest)
            declared = recovery.declared
            observed = (
                recovery.outcome,
                declared.set_id if declared else None,
                declared.set_digest if declared else None,
            )
            if (
                observed != expected
                or restarted.active() != declared
                or restarted.journal() is not None
            ):
                raise _CheckFailed("state_ambiguous")
            return recovery.outcome, stable_hash(restarted.checkpoint()["committed"])

        if not self._step("upgrade", "interrupted_attempt", _interrupted):
            return

        def _resume() -> str:
            assert self.installation is not None and self.plan is not None
            self.installation.transition(
                "upgrade",
                staged[0],
                distribution=self.plan.distribution,
                import_name=self.plan.import_name,
            )
            committed = self.installation.checkpoint()["committed"]
            if committed.count("upgrade") != 1:
                raise _CheckFailed("duplicate_commit")
            return self._verify_declared("upgrade")()

        if not self._step("upgrade", "resume_upgrade", _resume):
            return
        self._smoke_check("upgrade", "smoke_upgraded")

    def _smoke_check(self, phase: str, check: str) -> bool:
        _, smoke = self._smoke_active_or_fail(phase, check)
        if smoke is None:
            return False
        status = "passed" if smoke.reason == "ok" else "failed"
        return self._mark(
            phase, check, _check(status, smoke.reason, digest=smoke.digest)
        )

    def _phase_injected_failure(self) -> None:
        assert self.installation is not None and self.plan is not None

        def _inject() -> str:
            assert self.installation is not None and self.plan is not None
            active = self.installation.active()
            if active is None:
                raise _CheckFailed("declared_state_mismatch")
            self.installer.inject_disk_fault(
                self.installation.slot_path(active),
                import_name=self.plan.import_name,
            )
            self.disk_fault = "installed_file_modified"
            return active.set_digest

        if not self._step("injected_failure", "disk_fault_injected", _inject):
            return

        def _detected() -> tuple[str, None]:
            _, smoke = self._smoke_active()
            if smoke.installed_integrity != "installed_files_modified":
                raise _CheckFailed("fault_not_detected")
            return "fault_detected", None

        if not self._step("injected_failure", "fault_detected", _detected):
            return

        def _blocked() -> tuple[str, None]:
            allowed, _ = self._admission()
            if allowed:
                raise _CheckFailed("admission_not_blocked")
            return "admission_blocked", None

        self._step("injected_failure", "admission_blocked", _blocked)

    def _phase_rollback(self) -> None:
        assert self.installation is not None and self.plan is not None
        staged: list[StagedSet] = []

        def _verify() -> str:
            staged.append(self._staged("rollback"))
            return staged[0].digest

        if not self._step("rollback", "set_verified", _verify):
            return

        def _rollback() -> str | None:
            assert self.installation is not None and self.plan is not None
            result = self.installation.transition(
                "rollback",
                staged[0],
                distribution=self.plan.distribution,
                import_name=self.plan.import_name,
            )
            if result.outcome != "committed":
                raise _CheckFailed("duplicate_commit")
            return result.smoke.digest if result.smoke else None

        if not self._step("rollback", "rollback_release", _rollback):
            return
        if not self._smoke_check("rollback", "smoke_rolled_back"):
            return
        self._step("rollback", "declared_state", self._verify_declared("rollback"))

    def _phase_restore(self) -> None:
        assert self.plan is not None
        recovery = self.plan.sets.get("recovery")
        if recovery is not None:
            try:
                self._reverify(recovery)
            except _CheckFailed as exc:
                self._mark("restore", "backup_restore", _check("failed", exc.reason))
                return
        for hook in PHASES["restore"]:
            if not self._hook("restore", hook):
                return

    def _phase_continuity(self) -> None:
        assert self.installation is not None

        def _ledger() -> str:
            assert self.installation is not None
            recovery = self.installation.recover()
            if recovery.outcome != "clean":
                raise _CheckFailed("journal_open")
            committed = self.installation.checkpoint()["committed"]
            if len(set(committed)) != len(committed):
                raise _CheckFailed("duplicate_commit")
            if committed != list(TRANSACTIONS):
                raise _CheckFailed("declared_state_mismatch")
            return stable_hash(committed)

        if not self._step("continuity", "transaction_ledger", _ledger):
            return
        for hook in ("model_activation", "audit_continuity", "durable_work_resume"):
            if not self._hook("continuity", hook):
                return
        allowed, reason = self._admission()
        recovery_checks = [
            self.records["restore"].get(name, {}).get("status")
            for name in PHASES["restore"]
        ] + [
            self.records["continuity"].get(name, {}).get("status")
            for name in ("model_activation", "audit_continuity", "durable_work_resume")
        ]
        if not allowed:
            record = _check("failed", _reason_or(reason, "recovery_check_failed"))
        elif any(status != "passed" for status in recovery_checks):
            record = _check("not_available", "recovery_feature_unavailable")
        else:
            record = _check("passed", "admission_resumed")
        self._mark("continuity", "admission", record)


def _phase_status(checks: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    statuses = [check["status"] for check in checks]
    for check in checks:
        if check["status"] == "failed":
            return "failed", str(check["reason_code"])
    if all(status == "not_run" for status in statuses):
        return "not_run", "prior_phase_failed"
    if "not_run" in statuses:
        return "failed", "prior_check_failed"
    if "not_available" in statuses:
        return "incomplete", "recovery_feature_unavailable"
    return "passed", "ok"


def _not_run_phase(phase: str) -> dict[str, Any]:
    return {
        "id": phase,
        "status": "not_run",
        "reason_code": "prior_phase_failed",
        "checks": [
            {"id": check, **_check("not_run", "prior_phase_failed")}
            for check in PHASES[phase]
        ],
    }


def _empty_bundle_summary() -> dict[str, Any]:
    return {
        "manifest_digest": None,
        "artifact_count": None,
        "total_size_bytes": None,
        "sets": [],
        "migrations": [],
    }


def _bundle_summary(
    manifest: Mapping[str, Any] | None,
    manifest_digest: str | None,
    plan: LifecyclePlan | None,
) -> dict[str, Any]:
    summary = _empty_bundle_summary()
    summary["manifest_digest"] = manifest_digest
    if manifest is not None:
        for key in ("artifact_count", "total_size_bytes"):
            value = manifest.get(key)
            summary[key] = value if type(value) is int and value >= 0 else None
    if plan is not None:
        summary["sets"] = [
            {
                "id": set_id,
                "version": artifact_set.version,
                "digest": artifact_set.digest,
                "artifact_count": len(artifact_set.artifacts),
            }
            for set_id, artifact_set in plan.sets.items()
        ]
        summary["migrations"] = [
            {
                "id": migration.migration_id,
                "sequence": migration.sequence,
                "digest": "sha256:" + migration.sha256,
            }
            for migration in plan.migrations
        ]
    return summary


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_digest_or_none(path: Path) -> str | None:
    try:
        return "sha256:" + _file_sha256(path)
    except OSError:
        return None


def run_rehearsal_inside(
    bundle: Path,
    *,
    fault_point: str = DEFAULT_FAULT_POINT,
    hooks: Mapping[str, str] | None = None,
    installer: ReleaseInstaller | None = None,
    now: datetime | None = None,
    work_dir: Path | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run every lifecycle phase against a bundle and return value-free results.

    This is the workload ``scripts/airgap/rehearse_inside.py`` runs inside the
    no-egress boundary. All installation state lives in a private temporary
    directory that is removed afterwards.

    Args:
        bundle: Bundle directory or archive (read-only).
        fault_point: Where the first upgrade attempt is interrupted.
        hooks: Caller ``hook -> "module:callable"`` overrides.
        installer: Installer to use; defaults to :class:`PipVenvInstaller`.
        now: Clock for the validity window; defaults to the current UTC time.
        work_dir: Parent for the private state directory.
        verbose: Stream installer output to stderr (never to the result).

    Returns:
        ``bundle``, ``fault``, ``phases``, and ``final_state`` records.

    Raises:
        RehearsalError: ``fault_point_invalid`` for an unknown fault point.
    """

    if fault_point not in FAULT_POINTS:
        raise RehearsalError("fault_point_invalid")
    runner = _Rehearsal(
        hooks=resolve_hooks(hooks),
        installer=installer or PipVenvInstaller(verbose=verbose),
        fault_point=fault_point,
        now=now or datetime.now(timezone.utc),
    )
    with (
        tempfile.TemporaryDirectory(
            prefix="openmed-airgap-rehearsal-",
            dir=work_dir,
            ignore_cleanup_errors=True,
        ) as temporary,
        contextlib.ExitStack() as stack,
    ):
        try:
            root = stack.enter_context(open_bundle_root(bundle))
        except (OSError, ValueError, tarfile.TarError):
            runner.records["preflight"]["bundle_integrity"] = _check(
                "failed", "bundle_unreadable"
            )
            return runner.result()
        return runner.run(root, Path(temporary))


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _canonical(payload: Any) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


class RehearsalReport(AttestationReport):
    """Bounded, value-free, integrity-protected air-gap rehearsal report."""

    @property
    def status(self) -> str:
        """Return ``passed``, ``failed``, ``incomplete``, or ``skipped``."""

        return str(self._data["status"])

    @property
    def reason_code(self) -> str:
        """Return the controlled top-level reason code."""

        return str(self._data["reason_code"])

    @property
    def passed(self) -> bool:
        """Return whether every phase passed inside an isolated boundary."""

        return self.status == "passed"

    @property
    def exit_code(self) -> int:
        """Return the stable exit code; only a pass exits zero."""

        return STATUS_EXIT_CODES.get(self.status, EXIT_FAILED)

    def repro_hash_matches(self) -> bool:
        """Return whether the stable evidence matches ``repro_hash``."""

        payload = self.to_dict()
        expected = payload.pop("repro_hash", "")
        for name in REHEARSAL_VOLATILE_FIELDS:
            payload.pop(name, None)
        return expected == stable_hash(payload)

    def integrity_hash_matches(self) -> bool:
        """Return whether the payload matches ``integrity_hash``."""

        payload = self.to_dict()
        payload.pop("signature", None)
        expected = payload.pop("integrity_hash", "")
        return expected == stable_hash(payload)

    def signature_matches(self, key: bytes) -> bool:
        """Return whether the HMAC signature verifies under ``key``."""

        payload = self.to_dict()
        signature = payload.pop("signature", None)
        if not isinstance(signature, Mapping):
            return False
        if signature.get("algorithm") != SIGNATURE_ALGORITHM:
            return False
        expected = hmac.new(key, _canonical(payload), hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, str(signature.get("value", "")))

    def to_markdown(self) -> str:
        """Render a value-free Markdown summary."""

        payload = self.to_dict()
        boundary = payload["boundary"]
        signature = payload["signature"]
        lines = [
            "# OpenMed air-gap lifecycle rehearsal",
            "",
            f"- Status: `{payload['status']}` (`{payload['reason_code']}`)",
            f"- Failed phase: `{payload['failed_phase']}`",
            f"- Commit: `{payload['commit']}`",
            f"- OpenMed version: `{payload['openmed_version']}`",
            f"- Host kind: `{payload['host']['kind']}`",
            f"- Boundary: `{boundary['kind']}` (`{boundary['mechanism']}`)",
            f"- Boundary isolated: `{str(boundary['isolated']).lower()}`",
            f"- Manifest digest: `{(payload['bundle'] or {}).get('manifest_digest')}`",
            f"- Generated at: `{payload['generated_at']}`",
            f"- Repro hash: `{payload['repro_hash']}`",
            f"- Integrity hash: `{payload['integrity_hash']}`",
            "- Signature: "
            + (
                f"`{signature['algorithm']}` key `{signature['key_id']}`"
                if signature
                else "`none` (integrity hash only)"
            ),
            "",
            "## Phases",
            "",
            "| Phase | Check | Status | Reason | Digest |",
            "|---|---|---|---|---|",
        ]
        for phase in payload["phases"]:
            for check in phase["checks"]:
                lines.append(
                    f"| `{phase['id']}` | `{check['id']}` | `{check['status']}` "
                    f"| `{check['reason_code']}` | `{check['digest']}` |"
                )
        lines.extend(["", "## Lifecycle sets", ""])
        for item in (payload["bundle"] or {}).get("sets", []):
            lines.append(
                f"- `{item['id']}` `{item['version']}`: `{item['digest']}` "
                f"({item['artifact_count']} artifacts)"
            )
        counts = payload["counts"]
        lines.extend(
            [
                "",
                "## Network controls",
                "",
                "- Negative controls blocked before phases: "
                f"{counts['negative_controls']['blocked']}/"
                f"{counts['negative_controls']['total']}",
                "- Negative controls blocked after phases: "
                f"{counts['negative_controls_after']['blocked']}/"
                f"{counts['negative_controls_after']['total']}",
                "- Loopback positive controls reachable: "
                f"{counts['positive_controls']['reachable']}/"
                f"{counts['positive_controls']['total']}",
                "",
            ]
        )
        return "\n".join(lines)


def build_rehearsal_report(
    *,
    host_kind: str,
    requested_boundary: str,
    boundary_kind: str,
    evidence: Mapping[str, Any] | None = None,
    rehearsal: Mapping[str, Any] | None = None,
    skip_reason: str | None = None,
    runner_failure: str | None = None,
    commit: str | None = None,
    openmed_version: str | None = None,
    generated_at: datetime | str | None = None,
    key: bytes | None = None,
    key_id: str = "rehearsal",
) -> RehearsalReport:
    """Build, sign, and validate a rehearsal report.

    The status is derived from the evidence, never supplied: a rehearsal passes
    only when the boundary is an isolated OS mechanism, every negative control
    is blocked before and after the phases, every loopback control is
    reachable, and every phase passed. Any ``not_available`` recovery hook
    makes it ``incomplete``.

    Args:
        host_kind: ``linux``, ``darwin``, ``windows``, or ``other``.
        requested_boundary: ``auto``, ``netns``, or ``container``.
        boundary_kind: ``netns``, ``container``, or ``none``.
        evidence: In-boundary ``kernel``, ``interfaces``, ``routes``,
            ``negative_controls``, ``positive_controls``, and
            ``negative_controls_after``.
        rehearsal: The result of :func:`run_rehearsal_inside`.
        skip_reason: Controlled reason when no boundary ran.
        runner_failure: Controlled reason when the in-boundary runner failed.
        commit: Git commit; anything else is recorded as ``unknown``.
        openmed_version: Package version; defaults to the installed version.
        generated_at: Declared timestamp (volatile).
        key: Optional caller-held HMAC key; never stored.
        key_id: Identifier recorded next to the signature.

    Returns:
        The validated report.

    Raises:
        RehearsalReportError: With a controlled code for invalid input.
    """

    if host_kind not in _HOST_KINDS:
        raise RehearsalReportError("host_kind_invalid")
    if requested_boundary not in _REQUESTED_BOUNDARIES:
        raise RehearsalReportError("requested_boundary_invalid")
    if boundary_kind not in NO_EGRESS_BOUNDARY_MECHANISMS:
        raise RehearsalReportError("boundary_kind_invalid")
    if skip_reason is not None and skip_reason not in SKIP_REASONS:
        raise RehearsalReportError("skip_reason_invalid")
    if runner_failure is not None and runner_failure not in (
        NO_EGRESS_RUNNER_FAILURE_REASONS
    ):
        raise RehearsalReportError("runner_failure_invalid")
    if skip_reason is not None and (
        boundary_kind != "none" or evidence or rehearsal or runner_failure
    ):
        raise RehearsalReportError("skip_with_evidence")
    if skip_reason is None and boundary_kind == "none":
        raise RehearsalReportError("boundary_required")
    if key is not None and (len(key) < MIN_KEY_BYTES or not _KEY_ID_RE.match(key_id)):
        raise RehearsalReportError("signing_key_invalid")
    no_evidence = skip_reason is not None or runner_failure is not None
    source = {} if no_evidence else dict(evidence or {})
    try:
        kernel = source.get("kernel", "unknown")
        if kernel not in {"linux", "unknown"}:
            raise RehearsalReportError("kernel_invalid")
        interfaces = _count_mapping(
            source.get("interfaces"), ("total", "loopback_up", "non_loopback_up")
        )
        routes = _count_mapping(
            source.get("routes"), ("ipv4_non_loopback", "ipv6_non_loopback")
        )
        negatives = _probe_records(
            source.get("negative_controls", ()),
            NO_EGRESS_NEGATIVE_CONTROLS,
            _NEGATIVE_STATUSES,
        )
        negatives_after = _probe_records(
            source.get("negative_controls_after", ()),
            NO_EGRESS_NEGATIVE_CONTROLS,
            _NEGATIVE_STATUSES,
        )
        positives = _probe_records(
            source.get("positive_controls", ()),
            NO_EGRESS_POSITIVE_CONTROLS,
            _POSITIVE_STATUSES,
        )
    except (NoEgressAttestationError, TypeError, AttributeError) as exc:
        raise RehearsalReportError("boundary_evidence_invalid") from exc
    for record in (*negatives, *negatives_after):
        if record["status"] == "blocked" and record["reason_code"] not in (
            _BLOCKED_REASONS
        ):
            raise RehearsalReportError("blocked_reason_invalid")
    bundle: dict[str, Any] | None
    fault: dict[str, Any] | None
    phases: list[dict[str, Any]]
    final_state: dict[str, Any] | None
    if no_evidence:
        bundle, fault, phases, final_state = None, None, [], None
    else:
        if rehearsal is None:
            raise RehearsalReportError("rehearsal_result_invalid")
        bundle, fault, phases, final_state = _normalize_rehearsal(rehearsal)
    isolated = _isolated(boundary_kind, kernel, interfaces, routes)
    status, reason, failed_phase = _derive_status(
        isolated=isolated,
        negatives=negatives,
        negatives_after=negatives_after,
        positives=positives,
        phases=phases,
        skip_reason=skip_reason,
        runner_failure=runner_failure,
    )
    payload: dict[str, Any] = {
        "schema_version": REHEARSAL_SCHEMA_VERSION,
        "report_type": REHEARSAL_REPORT_TYPE,
        "generated_at": _iso_timestamp(generated_at),
        "volatile_fields": list(REHEARSAL_VOLATILE_FIELDS),
        "commit": _safe_commit(commit),
        "openmed_version": _safe_version(openmed_version),
        "status": status,
        "reason_code": reason,
        "failed_phase": failed_phase,
        "host": {"kind": host_kind},
        "boundary": {
            "requested": requested_boundary,
            "kind": boundary_kind,
            "mechanism": NO_EGRESS_BOUNDARY_MECHANISMS[boundary_kind],
            "kernel": kernel,
            "isolated": isolated,
            "interfaces": interfaces,
            "routes": routes,
        },
        "negative_controls": negatives,
        "negative_controls_after": negatives_after,
        "positive_controls": positives,
        "bundle": bundle,
        "fault": fault,
        "phases": phases,
        "final_state": final_state,
    }
    payload["counts"] = _counts(payload)
    stable = copy.deepcopy(payload)
    stable.pop("generated_at")
    payload["repro_hash"] = stable_hash(stable)
    payload["integrity_hash"] = stable_hash(payload)
    payload["signature"] = None
    if key is not None:
        unsigned = dict(payload)
        unsigned.pop("signature")
        payload["signature"] = {
            "algorithm": SIGNATURE_ALGORITHM,
            "key_id": key_id,
            "value": hmac.new(key, _canonical(unsigned), hashlib.sha256).hexdigest(),
        }
    return validate_rehearsal_report(payload, key=key)


def _timestamp_is_canonical(value: str) -> bool:
    try:
        return _iso_timestamp(value) == value
    except (TypeError, ValueError):
        return False


def _isolated(
    boundary_kind: str,
    kernel: str,
    interfaces: Mapping[str, int],
    routes: Mapping[str, int],
) -> bool:
    return bool(
        boundary_kind != "none"
        and kernel == "linux"
        and interfaces["loopback_up"] >= 1
        and interfaces["non_loopback_up"] == 0
        and routes["ipv4_non_loopback"] == 0
        and routes["ipv6_non_loopback"] == 0
    )


def _derive_status(
    *,
    isolated: bool,
    negatives: Sequence[Mapping[str, Any]],
    negatives_after: Sequence[Mapping[str, Any]],
    positives: Sequence[Mapping[str, Any]],
    phases: Sequence[Mapping[str, Any]],
    skip_reason: str | None,
    runner_failure: str | None,
) -> tuple[str, str, str | None]:
    if skip_reason is not None:
        return "skipped", skip_reason, None
    if runner_failure is not None:
        return "failed", runner_failure, None
    for records in (negatives, negatives_after):
        if any(record["status"] == "not_blocked" for record in records):
            return "failed", "egress_not_blocked", None
    for records in (negatives, negatives_after):
        statuses = {record["id"]: record["status"] for record in records}
        if set(statuses) != set(NO_EGRESS_NEGATIVE_CONTROLS) or any(
            value != "blocked" for value in statuses.values()
        ):
            return "failed", "negative_control_incomplete", None
    if not isolated:
        return "failed", "boundary_not_isolated", None
    statuses = {record["id"]: record["status"] for record in positives}
    if set(statuses) != set(NO_EGRESS_POSITIVE_CONTROLS) or any(
        value != "reachable" for value in statuses.values()
    ):
        return "failed", "positive_control_failed", None
    for phase in phases:
        if phase["status"] in {"failed", "not_run"}:
            return "failed", str(phase["reason_code"]), str(phase["id"])
    if any(phase["status"] == "incomplete" for phase in phases):
        return "incomplete", "recovery_feature_unavailable", None
    return "passed", "rehearsal_passed", None


def _status_counts(
    records: Sequence[Mapping[str, Any]], statuses: Sequence[str]
) -> dict[str, int]:
    counts = {"total": len(records)}
    for status in statuses:
        counts[status] = sum(1 for record in records if record["status"] == status)
    return counts


def _counts(payload: Mapping[str, Any]) -> dict[str, Any]:
    checks = [check for phase in payload["phases"] for check in phase["checks"]]
    return {
        "negative_controls": _status_counts(
            payload["negative_controls"], _NEGATIVE_STATUSES
        ),
        "negative_controls_after": _status_counts(
            payload["negative_controls_after"], _NEGATIVE_STATUSES
        ),
        "positive_controls": _status_counts(
            payload["positive_controls"], _POSITIVE_STATUSES
        ),
        "phases": _status_counts(payload["phases"], PHASE_STATUSES),
        "checks": _status_counts(checks, CHECK_STATUSES),
    }


def _normalize_rehearsal(
    raw: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any] | None]:
    """Strictly re-validate an in-boundary result before it enters a report."""

    if not isinstance(raw, Mapping) or set(raw) != {
        "bundle",
        "fault",
        "phases",
        "final_state",
    }:
        raise RehearsalReportError("rehearsal_result_invalid")
    return (
        _normalize_bundle(raw["bundle"]),
        _normalize_fault(raw["fault"]),
        _normalize_phases(raw["phases"]),
        _normalize_final_state(raw["final_state"]),
    )


def _invalid() -> RehearsalReportError:
    return RehearsalReportError("rehearsal_result_invalid")


def _optional_digest(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not _DIGEST_RE.match(value):
        raise _invalid()
    return value


def _count_or_none(value: Any) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise _invalid()
    return value


def _version(value: Any, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not _VERSION_RE.match(value):
        raise _invalid()
    return value


def _normalize_bundle(raw: Any) -> dict[str, Any]:
    keys = {
        "manifest_digest",
        "artifact_count",
        "total_size_bytes",
        "sets",
        "migrations",
    }
    if not isinstance(raw, Mapping) or set(raw) != keys:
        raise _invalid()
    sets = raw["sets"]
    migrations = raw["migrations"]
    if not isinstance(sets, list) or not isinstance(migrations, list):
        raise _invalid()
    if len(migrations) > 256:
        raise _invalid()
    order = list(SET_ROOTS)
    normalized_sets = []
    for item in sets:
        if not isinstance(item, Mapping) or set(item) != {
            "id",
            "version",
            "digest",
            "artifact_count",
        }:
            raise _invalid()
        if item["id"] not in SET_ROOTS:
            raise _invalid()
        normalized_sets.append(
            {
                "id": item["id"],
                "version": _version(item["version"], optional=item["id"] == "recovery"),
                "digest": _optional_digest(item["digest"]),
                "artifact_count": _count_or_none(item["artifact_count"]),
            }
        )
    ids = [item["id"] for item in normalized_sets]
    if ids != sorted(ids, key=order.index) or len(set(ids)) != len(ids):
        raise _invalid()
    normalized_migrations = []
    for item in migrations:
        if not isinstance(item, Mapping) or set(item) != {"id", "sequence", "digest"}:
            raise _invalid()
        if not isinstance(item["id"], str) or not _MIGRATION_ID_RE.match(item["id"]):
            raise _invalid()
        normalized_migrations.append(
            {
                "id": item["id"],
                "sequence": _count_or_none(item["sequence"]),
                "digest": _optional_digest(item["digest"]),
            }
        )
    return {
        "manifest_digest": _optional_digest(raw["manifest_digest"]),
        "artifact_count": _count_or_none(raw["artifact_count"]),
        "total_size_bytes": _count_or_none(raw["total_size_bytes"]),
        "sets": normalized_sets,
        "migrations": normalized_migrations,
    }


def _normalize_fault(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != {
        "upgrade_interruption",
        "recovery",
        "disk_fault",
    }:
        raise _invalid()
    if raw["upgrade_interruption"] not in FAULT_POINTS:
        raise _invalid()
    if raw["recovery"] is not None and raw["recovery"] not in RECOVERY_OUTCOMES:
        raise _invalid()
    if raw["disk_fault"] not in {None, "installed_file_modified"}:
        raise _invalid()
    return {
        "upgrade_interruption": raw["upgrade_interruption"],
        "recovery": raw["recovery"],
        "disk_fault": raw["disk_fault"],
    }


def _normalize_phases(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list) or [
        item.get("id") if isinstance(item, Mapping) else None for item in raw
    ] != list(PHASES):
        raise _invalid()
    phases = []
    for item in raw:
        if set(item) != {"id", "status", "reason_code", "checks"}:
            raise _invalid()
        checks_raw = item["checks"]
        if not isinstance(checks_raw, list) or [
            check.get("id") if isinstance(check, Mapping) else None
            for check in checks_raw
        ] != list(PHASES[item["id"]]):
            raise _invalid()
        checks = []
        for check in checks_raw:
            if set(check) != {"id", "status", "reason_code", "digest", "provider"}:
                raise _invalid()
            is_hook = check["id"] in RECOVERY_HOOKS
            provider = check["provider"]
            if (
                check["status"] not in CHECK_STATUSES
                or check["reason_code"] not in CHECK_REASON_CODES
                or (provider is not None and provider not in HOOK_PROVIDERS)
                or (provider is not None and not is_hook)
            ):
                raise _invalid()
            checks.append(
                {
                    "id": check["id"],
                    "status": check["status"],
                    "reason_code": check["reason_code"],
                    "digest": _optional_digest(check["digest"]),
                    "provider": provider,
                }
            )
        status, reason = _phase_status(checks)
        if item["status"] == "not_run" and status == "not_run":
            reason = item["reason_code"]
        if (item["status"], item["reason_code"]) != (status, reason) or (
            reason not in CHECK_REASON_CODES
        ):
            raise _invalid()
        phases.append(
            {
                "id": item["id"],
                "status": status,
                "reason_code": reason,
                "checks": checks,
            }
        )
    seen_failure = False
    for phase in phases:
        if seen_failure and phase["status"] != "not_run":
            raise _invalid()
        seen_failure = seen_failure or phase["status"] in {"failed", "not_run"}
    return phases


def _normalize_final_state(raw: Any) -> dict[str, Any] | None:
    if raw is None:
        return None
    keys = {
        "version",
        "set_id",
        "set_digest",
        "committed_transactions",
        "duplicate_commits",
        "journal_open",
    }
    if not isinstance(raw, Mapping) or set(raw) != keys:
        raise _invalid()
    if raw["set_id"] not in SET_ROOTS or not isinstance(raw["journal_open"], bool):
        raise _invalid()
    return {
        "version": _version(raw["version"]),
        "set_id": raw["set_id"],
        "set_digest": _optional_digest(raw["set_digest"]),
        "committed_transactions": _count_or_none(raw["committed_transactions"]),
        "duplicate_commits": _count_or_none(raw["duplicate_commits"]),
        "journal_open": raw["journal_open"],
    }


_REPORT_KEYS = frozenset(
    {
        "schema_version",
        "report_type",
        "generated_at",
        "volatile_fields",
        "commit",
        "openmed_version",
        "status",
        "reason_code",
        "failed_phase",
        "host",
        "boundary",
        "negative_controls",
        "negative_controls_after",
        "positive_controls",
        "bundle",
        "fault",
        "phases",
        "final_state",
        "counts",
        "repro_hash",
        "integrity_hash",
        "signature",
    }
)


def validate_rehearsal_report(
    payload: Mapping[str, Any],
    *,
    key: bytes | None = None,
    require_signature: bool = False,
) -> RehearsalReport:
    """Validate a rehearsal report offline.

    Recomputes the vocabulary, counts, derived status, both hashes, the size
    bound, and (with ``key``) the HMAC signature, so an edited report or a
    hand-written ``passed`` report does not validate.

    Args:
        payload: A decoded report.
        key: Caller-held HMAC key; required when the report must be signed.
        require_signature: Reject reports without a verifiable signature.

    Returns:
        The validated report.

    Raises:
        RehearsalReportError: With a controlled code when invalid.
    """

    if not isinstance(payload, Mapping) or set(payload) != _REPORT_KEYS:
        raise RehearsalReportError("report_schema_invalid")
    data = copy.deepcopy(dict(payload))
    if len(_canonical(data)) > MAX_REPORT_BYTES:
        raise RehearsalReportError("report_too_large")
    if (
        data["schema_version"] != REHEARSAL_SCHEMA_VERSION
        or data["report_type"] != REHEARSAL_REPORT_TYPE
        or data["volatile_fields"] != list(REHEARSAL_VOLATILE_FIELDS)
        or not isinstance(data["generated_at"], str)
        or not _timestamp_is_canonical(data["generated_at"])
        or data["commit"] != _safe_commit(data["commit"])
        or data["openmed_version"] != _safe_version(data["openmed_version"])
        or data["status"] not in REPORT_STATUSES
        or data["reason_code"] not in REPORT_REASON_CODES
        or not isinstance(data["host"], Mapping)
        or set(data["host"]) != {"kind"}
        or data["host"]["kind"] not in _HOST_KINDS
    ):
        raise RehearsalReportError("report_schema_invalid")
    boundary = data["boundary"]
    if not isinstance(boundary, Mapping) or set(boundary) != {
        "requested",
        "kind",
        "mechanism",
        "kernel",
        "isolated",
        "interfaces",
        "routes",
    }:
        raise RehearsalReportError("report_schema_invalid")
    if (
        boundary["requested"] not in _REQUESTED_BOUNDARIES
        or boundary["kind"] not in NO_EGRESS_BOUNDARY_MECHANISMS
        or boundary["mechanism"] != NO_EGRESS_BOUNDARY_MECHANISMS[boundary["kind"]]
        or boundary["kernel"] not in {"linux", "unknown"}
        or not isinstance(boundary["isolated"], bool)
    ):
        raise RehearsalReportError("report_schema_invalid")
    try:
        interfaces = _count_mapping(
            boundary["interfaces"], ("total", "loopback_up", "non_loopback_up")
        )
        routes = _count_mapping(
            boundary["routes"], ("ipv4_non_loopback", "ipv6_non_loopback")
        )
        controls = {
            name: _probe_records(data[name], vocabulary, statuses)
            for name, vocabulary, statuses in (
                ("negative_controls", NO_EGRESS_NEGATIVE_CONTROLS, _NEGATIVE_STATUSES),
                (
                    "negative_controls_after",
                    NO_EGRESS_NEGATIVE_CONTROLS,
                    _NEGATIVE_STATUSES,
                ),
                ("positive_controls", NO_EGRESS_POSITIVE_CONTROLS, _POSITIVE_STATUSES),
            )
        }
    except (NoEgressAttestationError, TypeError, AttributeError) as exc:
        raise RehearsalReportError("report_schema_invalid") from exc
    if interfaces != boundary["interfaces"] or routes != boundary["routes"]:
        raise RehearsalReportError("report_schema_invalid")
    for name, records in controls.items():
        if records != data[name]:
            raise RehearsalReportError("case_vocabulary_mismatch")
    for record in (
        *controls["negative_controls"],
        *controls["negative_controls_after"],
    ):
        if record["status"] == "blocked" and record["reason_code"] not in (
            _BLOCKED_REASONS
        ):
            raise RehearsalReportError("blocked_reason_invalid")
    isolated = _isolated(boundary["kind"], boundary["kernel"], interfaces, routes)
    if boundary["isolated"] is not isolated:
        raise RehearsalReportError("boundary_isolation_mismatch")

    status = data["status"]
    reason = data["reason_code"]
    no_evidence = status == "skipped" or reason in NO_EGRESS_RUNNER_FAILURE_REASONS
    expected: tuple[str, str, str | None]
    if no_evidence:
        if (
            data["phases"]
            or data["bundle"] is not None
            or data["fault"] is not None
            or data["final_state"] is not None
            or any(controls.values())
        ):
            raise RehearsalReportError("report_schema_invalid")
        if status == "skipped" and (
            reason not in SKIP_REASONS or boundary["kind"] != "none"
        ):
            raise RehearsalReportError("status_mismatch")
        if status != "skipped" and boundary["kind"] == "none":
            raise RehearsalReportError("status_mismatch")
        expected = (
            ("skipped", reason, None)
            if status == "skipped"
            else ("failed", reason, None)
        )
    else:
        if boundary["kind"] == "none":
            raise RehearsalReportError("status_mismatch")
        try:
            bundle, fault, phases, final_state = _normalize_rehearsal(
                {
                    "bundle": data["bundle"],
                    "fault": data["fault"],
                    "phases": data["phases"],
                    "final_state": data["final_state"],
                }
            )
        except RehearsalReportError as exc:
            raise RehearsalReportError("report_schema_invalid") from exc
        if (bundle, fault, phases, final_state) != (
            data["bundle"],
            data["fault"],
            data["phases"],
            data["final_state"],
        ):
            raise RehearsalReportError("report_schema_invalid")
        expected = _derive_status(
            isolated=isolated,
            negatives=controls["negative_controls"],
            negatives_after=controls["negative_controls_after"],
            positives=controls["positive_controls"],
            phases=phases,
            skip_reason=None,
            runner_failure=None,
        )
    if (status, reason, data["failed_phase"]) != expected:
        raise RehearsalReportError("status_mismatch")
    if data["counts"] != _counts(data):
        raise RehearsalReportError("counts_mismatch")
    signature = data["signature"]
    if signature is not None and (
        not isinstance(signature, Mapping)
        or set(signature) != {"algorithm", "key_id", "value"}
        or signature["algorithm"] != SIGNATURE_ALGORITHM
        or not isinstance(signature["key_id"], str)
        or not _KEY_ID_RE.match(signature["key_id"])
        or not isinstance(signature["value"], str)
        or not re.fullmatch(r"[0-9a-f]{64}", signature["value"])
    ):
        raise RehearsalReportError("signature_invalid")
    report = RehearsalReport(data)
    if not report.repro_hash_matches():
        raise RehearsalReportError("repro_hash_mismatch")
    if not report.integrity_hash_matches():
        raise RehearsalReportError("integrity_hash_mismatch")
    if require_signature and signature is None:
        raise RehearsalReportError("signature_missing")
    if require_signature and key is None:
        raise RehearsalReportError("signature_key_required")
    if key is not None and signature is not None and not report.signature_matches(key):
        raise RehearsalReportError("signature_mismatch")
    return report


# ---------------------------------------------------------------------------
# Host orchestration
# ---------------------------------------------------------------------------


def load_no_egress_gate(path: Path = GATE_PATH) -> ModuleType | None:
    """Load the no-egress gate runner from this checkout, if present.

    Args:
        path: The gate script.

    Returns:
        The loaded module, or ``None`` when the script is absent (for example
        in a wheel installation without the repository's ``scripts/``).
    """

    if not path.is_file():
        return None
    name = "openmed_airgap_no_egress_gate"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:  # pragma: no cover - a broken gate is treated as absent
        sys.modules.pop(name, None)
        return None
    return module


def _parseable_timestamp(value: str) -> bool:
    try:
        _iso_timestamp(value)
    except (TypeError, ValueError):
        return False
    return True


def _host_kind() -> str:
    system = sys.platform
    if system.startswith("linux"):
        return "linux"
    if system == "darwin":
        return "darwin"
    if system.startswith("win"):
        return "windows"
    return "other"


def rehearse(
    bundle: Path,
    *,
    boundary: str = "auto",
    container_image: str | None = None,
    container_tmpfs_size: str = DEFAULT_CONTAINER_TMPFS,
    hooks: Mapping[str, str] | None = None,
    fault_point: str = DEFAULT_FAULT_POINT,
    key: bytes | None = None,
    key_id: str = "rehearsal",
    generated_at: datetime | str | None = None,
    commit: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    verbose: bool = False,
    gate: ModuleType | None = None,
    inner_script: Path = INNER_SCRIPT,
) -> RehearsalReport:
    """Run the full lifecycle rehearsal inside the OS no-egress boundary.

    Args:
        bundle: Caller-supplied bundle directory or archive.
        boundary: ``auto``, ``netns``, or ``container``.
        container_image: Local image for container mode (never pulled);
            defaults to ``$OPENMED_NO_EGRESS_IMAGE``.
        container_tmpfs_size: Size cap for the container's writable ``/tmp``.
        hooks: Caller recovery hook overrides (``hook -> "module:callable"``),
            importable inside the boundary.
        fault_point: Where the first upgrade attempt is interrupted.
        key: Optional caller-held HMAC key for signing the report.
        key_id: Identifier recorded with the signature.
        generated_at: Declared report timestamp.
        commit: Git commit recorded in the report.
        timeout: Seconds before the in-boundary run is abandoned.
        verbose: Pass installer and runner diagnostics to stderr.
        gate: Preloaded no-egress gate module (tests).
        inner_script: In-boundary entry point inside this checkout.

    Returns:
        The validated, optionally signed rehearsal report. Hosts without a
        supported boundary yield ``skipped``, which is never a pass.

    Raises:
        RehearsalError: For an invalid boundary, fault point, or hook name.
        FileNotFoundError: If ``bundle`` does not exist.
    """

    if boundary not in _REQUESTED_BOUNDARIES:
        raise RehearsalError("boundary_invalid")
    if fault_point not in FAULT_POINTS:
        raise RehearsalError("fault_point_invalid")
    for name, spec in (hooks or {}).items():
        if name not in RECOVERY_HOOKS or not _HOOK_SPEC_RE.match(spec):
            raise RehearsalError("hook_spec_invalid")
    if generated_at is not None and not isinstance(generated_at, datetime):
        if not isinstance(generated_at, str) or not _parseable_timestamp(generated_at):
            raise RehearsalError("generated_at_invalid")
    bundle = bundle.expanduser().resolve(strict=True)
    common: dict[str, Any] = {
        "requested_boundary": boundary,
        "commit": commit,
        "generated_at": generated_at,
        "key": key,
        "key_id": key_id,
    }
    module = gate if gate is not None else load_no_egress_gate()
    if module is None or not inner_script.is_file():
        return build_rehearsal_report(
            host_kind=_host_kind(),
            boundary_kind="none",
            skip_reason="gate_unavailable",
            **common,
        )
    host = module.host_kind()
    image = container_image or os.environ.get(module.IMAGE_ENV_VAR) or None
    plan = module.select_boundary(boundary, host, image)
    if plan.skip_reason is not None:
        return build_rehearsal_report(
            host_kind=host,
            boundary_kind="none",
            skip_reason=plan.skip_reason,
            **common,
        )
    container_name = f"openmed-airgap-rehearsal-{uuid.uuid4().hex[:12]}"
    args = [
        "--boundary-kind",
        plan.kind,
        "--bundle",
        CONTAINER_BUNDLE if plan.kind == "container" else str(bundle),
        "--fault-point",
        fault_point,
    ]
    for name, spec in sorted((hooks or {}).items()):
        args.extend(["--hook", f"{name}={spec}"])
    if verbose:
        args.append("--verbose")
    try:
        command = module.boundary_command(
            plan,
            inner_script,
            args,
            container_name=container_name,
            mounts=[(bundle, CONTAINER_BUNDLE)],
            tmpfs_size=container_tmpfs_size,
        )
    except ValueError:
        return build_rehearsal_report(
            host_kind=host,
            boundary_kind=plan.kind,
            runner_failure="runner_failed",
            **common,
        )
    run = module.run_in_boundary(
        plan,
        command,
        container_name=container_name,
        verbose=verbose,
        timeout=timeout,
    )
    failure = run.failure
    result: Mapping[str, Any] | None = None
    if failure is None:
        outcome = module.parse_inner_output(
            run.stdout, run.returncode, required=INNER_RESULT_KEYS
        )
        failure, result = outcome.failure, outcome.result
    if failure is not None or result is None:
        return build_rehearsal_report(
            host_kind=host,
            boundary_kind=plan.kind,
            runner_failure=failure or "runner_failed",
            **common,
        )
    try:
        return build_rehearsal_report(
            host_kind=host,
            boundary_kind=plan.kind,
            evidence=result,
            rehearsal=result["rehearsal"],
            **common,
        )
    except (RehearsalReportError, KeyError, TypeError, AttributeError):
        return build_rehearsal_report(
            host_kind=host,
            boundary_kind=plan.kind,
            runner_failure="runner_result_invalid",
            **common,
        )


# ---------------------------------------------------------------------------
# CLI handlers (registered by ``openmed.cli.airgap``)
# ---------------------------------------------------------------------------


def read_key_file(path: Path) -> bytes:
    """Read a caller-held HMAC key; trailing newlines are ignored.

    Raises:
        OSError: If the file cannot be read.
        RehearsalError: ``signing_key_too_short`` below 16 bytes.
    """

    key = path.expanduser().read_bytes().rstrip(b"\r\n")
    if len(key) < MIN_KEY_BYTES:
        raise RehearsalError("signing_key_too_short")
    return key


def handle_rehearse(args: argparse.Namespace) -> int:
    """Run ``openmed airgap rehearse`` and return its stable exit code."""

    try:
        key = read_key_file(args.key_file) if args.key_file else None
    except RehearsalError as exc:
        sys.stderr.write(f"airgap rehearse: {exc.code}\n")
        return EXIT_USAGE
    except OSError:
        sys.stderr.write("airgap rehearse: signing_key_unreadable\n")
        return EXIT_USAGE
    try:
        hooks = parse_hook_specs(args.hooks or ())
        if not _KEY_ID_RE.match(args.key_id):
            raise RehearsalError("signing_key_id_invalid")
        report = rehearse(
            args.bundle,
            boundary=args.boundary,
            container_image=args.container_image,
            container_tmpfs_size=args.container_tmpfs_size,
            hooks=hooks,
            fault_point=args.inject_fault,
            key=key,
            key_id=args.key_id,
            generated_at=args.generated_at,
            commit=args.commit,
            timeout=args.timeout,
            verbose=args.verbose,
        )
    except RehearsalError as exc:
        sys.stderr.write(f"airgap rehearse: {exc.code}\n")
        return EXIT_USAGE
    except OSError:
        sys.stderr.write("airgap rehearse: bundle_unreadable\n")
        return EXIT_USAGE
    text = report.to_json(indent=2) + "\n"
    if args.output is None:
        sys.stdout.write(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(report.to_markdown(), encoding="utf-8")
    payload = report.to_dict()
    counts = payload["counts"]
    sys.stderr.write(
        "airgap rehearse: "
        f"status={payload['status']} reason={payload['reason_code']} "
        f"boundary={payload['boundary']['kind']} "
        f"phases_passed={counts['phases']['passed']}/{counts['phases']['total']} "
        f"negative_blocked={counts['negative_controls']['blocked']}/"
        f"{counts['negative_controls']['total']} "
        f"signed={'yes' if payload['signature'] else 'no'}\n"
    )
    return report.exit_code


def handle_verify_report(args: argparse.Namespace) -> int:
    """Run ``openmed airgap verify-report`` and return its exit code."""

    try:
        key = read_key_file(args.key_file) if args.key_file else None
    except RehearsalError as exc:
        sys.stderr.write(f"airgap verify-report: {exc.code}\n")
        return EXIT_USAGE
    except OSError:
        sys.stderr.write("airgap verify-report: signing_key_unreadable\n")
        return EXIT_USAGE
    try:
        payload = json.loads(args.report.read_text(encoding="utf-8"))
        report = validate_rehearsal_report(
            payload, key=key, require_signature=args.require_signature
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        sys.stderr.write("airgap verify-report: report_unreadable\n")
        return EXIT_FAILED
    except RehearsalReportError as exc:
        sys.stderr.write(f"airgap verify-report: {exc.code}\n")
        return EXIT_FAILED
    signature = (
        "verified"
        if key is not None and report.to_dict()["signature"]
        else ("present" if report.to_dict()["signature"] else "absent")
    )
    sys.stderr.write(
        f"airgap verify-report: valid status={report.status} "
        f"reason={report.reason_code} signature={signature}\n"
    )
    return report.exit_code


__all__ = [
    "BUILTIN_HOOKS",
    "DEFAULT_FAULT_POINT",
    "EXIT_FAILED",
    "EXIT_INCOMPLETE",
    "EXIT_PASSED",
    "EXIT_SKIPPED",
    "EXIT_USAGE",
    "FAULT_POINTS",
    "PHASES",
    "RECOVERY_HOOKS",
    "REHEARSAL_REPORT_TYPE",
    "REHEARSAL_SCHEMA_VERSION",
    "HookContext",
    "HookResult",
    "InjectedFault",
    "LifecycleInstallation",
    "PipVenvInstaller",
    "RehearsalError",
    "RehearsalReport",
    "RehearsalReportError",
    "ReleaseInstaller",
    "ReleaseRecord",
    "SmokeResult",
    "StagedSet",
    "build_rehearsal_report",
    "handle_rehearse",
    "handle_verify_report",
    "load_no_egress_gate",
    "parse_hook_specs",
    "read_key_file",
    "rehearse",
    "resolve_hooks",
    "run_hook",
    "run_rehearsal_inside",
    "validate_rehearsal_report",
]
