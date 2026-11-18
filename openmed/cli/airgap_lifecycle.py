"""Versioned lifecycle contract carried by an OpenMed air-gap bundle manifest.

An air-gap kit built by ``openmed airgap bundle`` can carry, next to the release
wheelhouse in ``wheels/``, every input an offline site needs to move between
releases without reaching any external service:

* ``lifecycle/upgrade/``: the next release's complete wheel/sdist set;
* ``lifecycle/rollback/``: the previous release's artifacts, so a failed
  upgrade can be undone even after an in-place install overwrote the old one;
* ``lifecycle/migrations/``: ordered migration assets for the upgrade edge;
* ``lifecycle/recovery/``: recovery assets (tools, backup fixtures) for the
  restore phase.

Every file is still listed with its size and SHA-256 digest in the manifest's
``artifacts`` inventory. The ``lifecycle`` section adds, per set, the declared
version, the files that belong to it, and a set digest; plus the compatibility
window, the migration chain, and an optional validity window. The rehearsal
(:mod:`openmed.cli.airgap_rehearsal`) checks all of it before mutating an
installation and reports only controlled reason codes, versions, and digests.
"""

from __future__ import annotations

import re
import shutil
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from ..core.audit import stable_hash

LIFECYCLE_SCHEMA_VERSION = 1
DISTRIBUTION = "openmed"
IMPORT_NAME = "openmed"
RELEASE_ROOT = "wheels"
UPGRADE_ROOT = "lifecycle/upgrade"
ROLLBACK_ROOT = "lifecycle/rollback"
MIGRATIONS_ROOT = "lifecycle/migrations"
RECOVERY_ROOT = "lifecycle/recovery"
SET_ROOTS: Mapping[str, str] = {
    "release": RELEASE_ROOT,
    "upgrade": UPGRADE_ROOT,
    "rollback": ROLLBACK_ROOT,
    "recovery": RECOVERY_ROOT,
}
"""Lifecycle set identifier to its bundle-relative root directory."""

INSTALL_SETS = ("release", "upgrade", "rollback")
"""Sets that hold installable wheels or sdists."""

INSTALLABLE_SUFFIXES = (".whl", ".tar.gz", ".zip")
MAX_SET_ARTIFACTS = 4096
MAX_MIGRATIONS = 256

LIFECYCLE_REASON_CODES = frozenset(
    {
        "ok",
        "not_required",
        "lifecycle_missing",
        "lifecycle_invalid",
        "set_missing",
        "set_digest_mismatch",
        "set_version_mismatch",
        "version_order_invalid",
        "upgrade_incompatible",
        "rollback_target_mismatch",
        "migration_order_invalid",
        "target_incompatible",
        "artifacts_expired",
        "artifacts_not_yet_valid",
        "prior_check_failed",
    }
)
"""Controlled reason codes produced by :func:`validate_lifecycle`."""

_VERSION_PATTERN = re.compile(
    r"^(\d+(?:\.\d+)*)(?:(a|b|rc)(\d+))?(?:\.post(\d+))?(?:\.dev(\d+))?$"
)
_MIGRATION_NAME = re.compile(r"^(\d{4})-[A-Za-z0-9][A-Za-z0-9._-]*$")
_DIGEST_HEX = re.compile(r"^[0-9a-f]{64}$")
_PRE_RELEASE_ORDER = {"a": 0, "b": 1, "rc": 2}
_UNBOUNDED = 2**63


class LifecycleError(ValueError):
    """Raised with one controlled reason code; never carries paths or values."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class LifecycleInputs:
    """Caller-supplied directories that extend a bundle with lifecycle inputs.

    Nothing is downloaded or substituted: each directory is copied verbatim
    into the bundle and every file is digested.

    Attributes:
        upgrade: Directory with the next release's complete wheel/sdist set.
        rollback: Directory with the previous release's artifacts, used to
            return to the pre-upgrade release.
        upgrade_from: Release versions the upgrade supports; defaults to the
            bundled release version.
        migrations: Optional directory of ``NNNN-<name>`` migration assets,
            applied in order on the release-to-upgrade edge.
        recovery: Optional directory of recovery assets for the restore phase.
        not_before: Optional start of the kit's validity window.
        not_after: Optional expiry of the kit's validity window.
    """

    upgrade: Path
    rollback: Path
    upgrade_from: Sequence[str] = ()
    migrations: Path | None = None
    recovery: Path | None = None
    not_before: datetime | str | None = None
    not_after: datetime | str | None = None


@dataclass(frozen=True)
class ArtifactSet:
    """One verified lifecycle set.

    Attributes:
        set_id: ``release``, ``upgrade``, ``rollback``, or ``recovery``.
        version: Declared release version; ``None`` for recovery assets.
        root: Bundle-relative directory that holds the set.
        artifacts: ``(bundle_relative_path, sha256_hex)`` pairs.
        digest: ``sha256:`` digest over the set's relative paths and digests.
        from_versions: Versions an upgrade set supports upgrading from.
    """

    set_id: str
    version: str | None
    root: str
    artifacts: tuple[tuple[str, str], ...]
    digest: str
    from_versions: tuple[str, ...] = ()


@dataclass(frozen=True)
class Migration:
    """One ordered migration asset on the release-to-upgrade edge."""

    migration_id: str
    sequence: int
    from_version: str
    to_version: str
    path: str
    sha256: str


@dataclass(frozen=True)
class LifecyclePlan:
    """The parsed, cross-checked lifecycle section of a bundle manifest."""

    distribution: str
    import_name: str
    sets: Mapping[str, ArtifactSet]
    migrations: tuple[Migration, ...]
    not_before: datetime | None
    not_after: datetime | None


@dataclass(frozen=True)
class LifecycleCheck:
    """A value-free preflight outcome: identifier, status, and reason code."""

    check_id: str
    status: str
    reason_code: str


@dataclass(frozen=True)
class LifecycleValidation:
    """Preflight outcome for the lifecycle contract of a bundle."""

    checks: tuple[LifecycleCheck, ...]
    plan: LifecyclePlan | None = field(default=None)

    @property
    def passed(self) -> bool:
        """Return whether every lifecycle check passed."""

        return all(check.status == "passed" for check in self.checks)


def version_key(value: str) -> tuple[Any, ...]:
    """Return a sort key for a PEP 440 release, pre, post, or dev version.

    Args:
        value: A version such as ``4.0.0``, ``4.0.0rc1``, or ``4.0.1.dev2``.

    Returns:
        A tuple that orders versions the way installers do.

    Raises:
        LifecycleError: ``lifecycle_invalid`` for an unsupported version.
    """

    match = _VERSION_PATTERN.match(value.strip().lower()) if value else None
    if match is None:
        raise LifecycleError("lifecycle_invalid")
    release = [int(part) for part in match.group(1).split(".")]
    while len(release) > 1 and release[-1] == 0:
        release.pop()
    pre, pre_number, post, dev = match.group(2, 3, 4, 5)
    if pre is not None:
        pre_key = (_PRE_RELEASE_ORDER[pre], int(pre_number))
    elif dev is not None and post is None:
        pre_key = (-1, 0)
    else:
        pre_key = (3, 0)
    return (
        tuple(release),
        pre_key,
        int(post) if post is not None else -1,
        int(dev) if dev is not None else _UNBOUNDED,
    )


def normalize_distribution(name: str) -> str:
    """Return the PEP 503 normalized form of a distribution name."""

    return re.sub(r"[-_.]+", "-", name).lower()


def artifact_identity(filename: str) -> tuple[str, str] | None:
    """Return ``(normalized_name, version)`` parsed from a wheel/sdist name."""

    if filename.endswith(".whl"):
        parts = filename[: -len(".whl")].split("-")
        if len(parts) not in {5, 6}:
            return None
        return normalize_distribution(parts[0]), parts[1]
    for suffix in (".tar.gz", ".zip"):
        if filename.endswith(suffix):
            stem = filename[: -len(suffix)]
            if "-" not in stem:
                return None
            name, version = stem.rsplit("-", 1)
            return normalize_distribution(name), version
    return None


def is_pure_wheel(filename: str) -> bool:
    """Return whether a wheel is interpreter- and platform-independent."""

    if not filename.endswith(".whl"):
        return True  # sdists are built for the running interpreter
    parts = filename[: -len(".whl")].split("-")
    if len(parts) not in {5, 6}:
        return False
    python_tag, abi_tag, platform_tag = parts[-3:]
    return (
        abi_tag == "none"
        and platform_tag == "any"
        and all(tag.startswith("py") for tag in python_tag.split("."))
    )


def set_digest(root: str, records: Sequence[tuple[str, str]]) -> str:
    """Return the ``sha256:`` digest of a set's relative paths and digests.

    Paths are made relative to the set root, so identical artifacts held under
    different roots (for example a rollback copy of the release) share a
    digest.

    Args:
        root: The set's bundle-relative root directory.
        records: ``(bundle_relative_path, sha256_hex)`` pairs.

    Returns:
        The set digest.
    """

    prefix = root + "/"
    rows = sorted(
        [path[len(prefix) :] if path.startswith(prefix) else path, digest]
        for path, digest in records
    )
    return stable_hash(rows)


def stage_lifecycle_inputs(bundle_root: Path, inputs: LifecycleInputs) -> None:
    """Copy caller-supplied lifecycle directories into a bundle being built.

    Args:
        bundle_root: The bundle directory, before artifacts are inventoried.
        inputs: The caller-supplied lifecycle directories.

    Raises:
        ValueError: If a directory is missing, empty, holds a symlink, holds a
            nested directory where a flat set is required, or a wheel set holds
            a file that is neither a wheel nor an sdist.
    """

    _copy_inputs(inputs.upgrade, bundle_root / UPGRADE_ROOT, kind="wheel_set")
    _copy_inputs(inputs.rollback, bundle_root / ROLLBACK_ROOT, kind="wheel_set")
    if inputs.migrations is not None:
        _copy_inputs(inputs.migrations, bundle_root / MIGRATIONS_ROOT, kind="flat")
    if inputs.recovery is not None:
        _copy_inputs(inputs.recovery, bundle_root / RECOVERY_ROOT, kind="tree")


def build_lifecycle_section(
    artifacts: Sequence[Mapping[str, Any]],
    *,
    release_version: str,
    inputs: LifecycleInputs,
) -> dict[str, Any]:
    """Return the manifest ``lifecycle`` section for staged inputs.

    Args:
        artifacts: The manifest artifact inventory (``path`` and ``sha256``).
        release_version: The version of the release in ``wheels/``.
        inputs: The lifecycle inputs staged with :func:`stage_lifecycle_inputs`.

    Returns:
        A JSON-compatible lifecycle section.

    Raises:
        LifecycleError: If a set has no artifact for its distribution.
    """

    digests = {str(item["path"]): str(item["sha256"]) for item in artifacts}

    def _set(set_id: str, version: str | None) -> dict[str, Any]:
        root = SET_ROOTS[set_id]
        records = [
            (path, digests[path])
            for path in sorted(digests)
            if path.startswith(root + "/")
        ]
        section: dict[str, Any] = {
            "root": root,
            "artifacts": [{"path": path, "sha256": sha} for path, sha in records],
            "digest": set_digest(root, records),
        }
        if version is not None:
            section["version"] = version
        return section

    upgrade_version = _set_version(digests, UPGRADE_ROOT)
    rollback_version = _set_version(digests, ROLLBACK_ROOT)
    sets: dict[str, Any] = {
        "release": _set("release", release_version),
        "upgrade": _set("upgrade", upgrade_version),
        "rollback": _set("rollback", rollback_version),
    }
    sets["upgrade"]["from_versions"] = sorted(
        set(inputs.upgrade_from) or {release_version}, key=version_key
    )
    if inputs.recovery is not None:
        sets["recovery"] = _set("recovery", None)
    migrations = []
    for path in sorted(digests):
        if not path.startswith(MIGRATIONS_ROOT + "/"):
            continue
        name = PurePosixPath(path).name
        match = _MIGRATION_NAME.match(name)
        if match is None:
            raise LifecycleError("migration_order_invalid")
        migrations.append(
            {
                "id": match.group(1),
                "sequence": int(match.group(1)),
                "from_version": release_version,
                "to_version": upgrade_version,
                "path": path,
                "sha256": digests[path],
            }
        )
    return {
        "schema_version": LIFECYCLE_SCHEMA_VERSION,
        "distribution": DISTRIBUTION,
        "import_name": IMPORT_NAME,
        "sets": sets,
        "migrations": migrations,
        "validity": {
            "not_before": _iso_or_none(inputs.not_before),
            "not_after": _iso_or_none(inputs.not_after),
        },
    }


def validate_lifecycle(
    manifest: Mapping[str, Any],
    *,
    now: datetime | None = None,
    python_version: str | None = None,
) -> LifecycleValidation:
    """Check a manifest's lifecycle contract before any installation changes.

    Three checks run in order: ``lifecycle_contract`` (structure, set
    membership, set digests, declared versions against artifact names),
    ``compatibility`` (version order, upgrade window, rollback target,
    migration chain, target interpreter), and ``validity_window`` (expiry).
    A later check is ``not_run`` when the contract itself is invalid.

    Args:
        manifest: A decoded ``bundle-manifest.json``.
        now: Clock for the validity window; defaults to the current UTC time.
        python_version: Interpreter ``major.minor`` that will install the sets;
            defaults to the running interpreter.

    Returns:
        Value-free checks and, when the contract is valid, the parsed plan.
    """

    current = now or datetime.now(timezone.utc)
    running = python_version or f"{sys.version_info.major}.{sys.version_info.minor}"
    try:
        plan = _parse_plan(manifest)
    except LifecycleError as exc:
        return LifecycleValidation(
            checks=(
                LifecycleCheck("lifecycle_contract", "failed", exc.code),
                LifecycleCheck("compatibility", "not_run", "prior_check_failed"),
                LifecycleCheck("validity_window", "not_run", "prior_check_failed"),
            )
        )
    contract = LifecycleCheck("lifecycle_contract", "passed", "ok")
    compatibility_reason = _compatibility_reason(manifest, plan, running)
    compatibility = LifecycleCheck(
        "compatibility",
        "passed" if compatibility_reason == "ok" else "failed",
        compatibility_reason,
    )
    validity_reason = _validity_reason(plan, current)
    validity = LifecycleCheck(
        "validity_window",
        "failed" if validity_reason.startswith("artifacts_") else "passed",
        validity_reason,
    )
    return LifecycleValidation(checks=(contract, compatibility, validity), plan=plan)


def _parse_plan(manifest: Mapping[str, Any]) -> LifecyclePlan:
    section = manifest.get("lifecycle")
    if section is None:
        raise LifecycleError("lifecycle_missing")
    if not isinstance(section, Mapping):
        raise LifecycleError("lifecycle_invalid")
    if section.get("schema_version") != LIFECYCLE_SCHEMA_VERSION:
        raise LifecycleError("lifecycle_invalid")
    if section.get("distribution") != DISTRIBUTION or (
        section.get("import_name") != IMPORT_NAME
    ):
        raise LifecycleError("lifecycle_invalid")
    inventory = _inventory(manifest)
    raw_sets = section.get("sets")
    if not isinstance(raw_sets, Mapping):
        raise LifecycleError("lifecycle_invalid")
    if set(raw_sets) - set(SET_ROOTS):
        raise LifecycleError("lifecycle_invalid")
    sets: dict[str, ArtifactSet] = {}
    for set_id in (*INSTALL_SETS, "recovery"):
        raw = raw_sets.get(set_id)
        if raw is None:
            if set_id == "recovery":
                if any(path.startswith(RECOVERY_ROOT + "/") for path in inventory):
                    raise LifecycleError("lifecycle_invalid")
                continue
            raise LifecycleError("set_missing")
        sets[set_id] = _parse_set(set_id, raw, inventory)
    if manifest.get("openmed_version") != sets["release"].version:
        raise LifecycleError("lifecycle_invalid")
    migrations = _parse_migrations(section.get("migrations"), inventory)
    validity = section.get("validity")
    if not isinstance(validity, Mapping) or set(validity) - {
        "not_before",
        "not_after",
    }:
        raise LifecycleError("lifecycle_invalid")
    return LifecyclePlan(
        distribution=DISTRIBUTION,
        import_name=IMPORT_NAME,
        sets=sets,
        migrations=migrations,
        not_before=_parse_time(validity.get("not_before")),
        not_after=_parse_time(validity.get("not_after")),
    )


def _inventory(manifest: Mapping[str, Any]) -> dict[str, str]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise LifecycleError("lifecycle_invalid")
    inventory: dict[str, str] = {}
    for item in artifacts:
        if not isinstance(item, Mapping):
            raise LifecycleError("lifecycle_invalid")
        path, sha = item.get("path"), item.get("sha256")
        if isinstance(path, str) and isinstance(sha, str):
            inventory[path] = sha
    return inventory


def _parse_set(set_id: str, raw: Any, inventory: Mapping[str, str]) -> ArtifactSet:
    root = SET_ROOTS[set_id]
    if not isinstance(raw, Mapping) or raw.get("root") != root:
        raise LifecycleError("lifecycle_invalid")
    entries = raw.get("artifacts")
    if not isinstance(entries, list) or not entries:
        raise LifecycleError("set_missing")
    if len(entries) > MAX_SET_ARTIFACTS:
        raise LifecycleError("lifecycle_invalid")
    records: list[tuple[str, str]] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise LifecycleError("lifecycle_invalid")
        path, sha = entry.get("path"), entry.get("sha256")
        if (
            not isinstance(path, str)
            or not isinstance(sha, str)
            or not _DIGEST_HEX.match(sha)
            or not path.startswith(root + "/")
        ):
            raise LifecycleError("lifecycle_invalid")
        if inventory.get(path) != sha:
            # The set names a file the inventory does not vouch for.
            raise LifecycleError("set_digest_mismatch")
        records.append((path, sha))
    declared = {path for path, _ in records}
    if len(declared) != len(records):
        raise LifecycleError("lifecycle_invalid")
    undeclared = {path for path in inventory if path.startswith(root + "/")} - declared
    if undeclared:
        raise LifecycleError("set_digest_mismatch")
    if raw.get("digest") != set_digest(root, records):
        raise LifecycleError("set_digest_mismatch")
    version: str | None = None
    from_versions: tuple[str, ...] = ()
    if set_id in INSTALL_SETS:
        version = raw.get("version")
        if not isinstance(version, str):
            raise LifecycleError("lifecycle_invalid")
        version_key(version)
        _require_distribution(records, version)
        if set_id == "upgrade":
            raw_from = raw.get("from_versions")
            if (
                not isinstance(raw_from, list)
                or not raw_from
                or not all(isinstance(item, str) for item in raw_from)
            ):
                raise LifecycleError("lifecycle_invalid")
            for item in raw_from:
                version_key(item)
            from_versions = tuple(raw_from)
    elif "version" in raw:
        raise LifecycleError("lifecycle_invalid")
    return ArtifactSet(
        set_id=set_id,
        version=version,
        root=root,
        artifacts=tuple(records),
        digest=str(raw["digest"]),
        from_versions=from_versions,
    )


def _require_distribution(records: Sequence[tuple[str, str]], version: str) -> None:
    found: list[str] = []
    for path, _ in records:
        name = PurePosixPath(path).name
        if not name.endswith(INSTALLABLE_SUFFIXES):
            raise LifecycleError("lifecycle_invalid")
        identity = artifact_identity(name)
        if identity is None:
            raise LifecycleError("lifecycle_invalid")
        if identity[0] == DISTRIBUTION:
            found.append(identity[1])
    if len(found) != 1 or version_key(found[0]) != version_key(version):
        raise LifecycleError("set_version_mismatch")


def _parse_migrations(raw: Any, inventory: Mapping[str, str]) -> tuple[Migration, ...]:
    if not isinstance(raw, list) or len(raw) > MAX_MIGRATIONS:
        raise LifecycleError("lifecycle_invalid")
    migrations: list[Migration] = []
    for entry in raw:
        if not isinstance(entry, Mapping):
            raise LifecycleError("lifecycle_invalid")
        path, sha = entry.get("path"), entry.get("sha256")
        sequence = entry.get("sequence")
        identifier = entry.get("id")
        from_version, to_version = entry.get("from_version"), entry.get("to_version")
        if (
            not isinstance(path, str)
            or not path.startswith(MIGRATIONS_ROOT + "/")
            or not isinstance(sha, str)
            or type(sequence) is not int
            or not isinstance(identifier, str)
            or not isinstance(from_version, str)
            or not isinstance(to_version, str)
        ):
            raise LifecycleError("lifecycle_invalid")
        if inventory.get(path) != sha:
            raise LifecycleError("set_digest_mismatch")
        version_key(from_version)
        version_key(to_version)
        migrations.append(
            Migration(identifier, sequence, from_version, to_version, path, sha)
        )
    declared = {migration.path for migration in migrations}
    present = {path for path in inventory if path.startswith(MIGRATIONS_ROOT + "/")}
    if declared != present or len(declared) != len(migrations):
        raise LifecycleError("set_digest_mismatch")
    return tuple(migrations)


def _compatibility_reason(
    manifest: Mapping[str, Any], plan: LifecyclePlan, running: str
) -> str:
    release = plan.sets["release"]
    upgrade = plan.sets["upgrade"]
    rollback = plan.sets["rollback"]
    assert release.version and upgrade.version and rollback.version
    if version_key(upgrade.version) <= version_key(release.version):
        return "version_order_invalid"
    if version_key(release.version) not in {
        version_key(item) for item in upgrade.from_versions
    }:
        return "upgrade_incompatible"
    if version_key(rollback.version) != version_key(release.version):
        return "rollback_target_mismatch"
    expected_sequence = 1
    for migration in plan.migrations:
        if (
            migration.sequence != expected_sequence
            or migration.migration_id != f"{migration.sequence:04d}"
            or version_key(migration.from_version) != version_key(release.version)
            or version_key(migration.to_version) != version_key(upgrade.version)
        ):
            return "migration_order_invalid"
        expected_sequence += 1
    target = manifest.get("target")
    target_python = (
        target.get("python_version") if isinstance(target, Mapping) else None
    )
    for set_id in INSTALL_SETS:
        for path, _ in plan.sets[set_id].artifacts:
            if not is_pure_wheel(PurePosixPath(path).name) and target_python != running:
                return "target_incompatible"
    return "ok"


def _validity_reason(plan: LifecyclePlan, now: datetime) -> str:
    if plan.not_before is None and plan.not_after is None:
        return "not_required"
    if plan.not_before is not None and now < plan.not_before:
        return "artifacts_not_yet_valid"
    if plan.not_after is not None and now > plan.not_after:
        return "artifacts_expired"
    return "ok"


def _set_version(digests: Mapping[str, str], root: str) -> str:
    versions = [
        identity[1]
        for path in digests
        if path.startswith(root + "/")
        and (identity := artifact_identity(PurePosixPath(path).name)) is not None
        and identity[0] == DISTRIBUTION
    ]
    if len(versions) != 1:
        raise LifecycleError("set_version_mismatch")
    version_key(versions[0])
    return versions[0]


def _copy_inputs(source: Path, destination: Path, *, kind: str) -> None:
    source = source.expanduser()
    if source.is_symlink() or not source.is_dir():
        raise ValueError("lifecycle input must be an existing directory")
    files: list[Path] = []
    for path in sorted(source.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise ValueError("lifecycle inputs must not contain symlinks")
        if path.is_dir():
            if kind != "tree":
                raise ValueError("lifecycle wheel and migration sets must be flat")
            continue
        if not path.is_file():
            raise ValueError("lifecycle inputs must be regular files")
        if kind == "wheel_set" and not path.name.endswith(INSTALLABLE_SUFFIXES):
            raise ValueError("lifecycle wheel sets may hold only wheels or sdists")
        if kind == "flat" and _MIGRATION_NAME.match(path.name) is None:
            raise ValueError("migration assets must be named NNNN-<name>")
        files.append(path)
    if not files:
        raise ValueError("lifecycle input directory is empty")
    for path in files:
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)


def _iso_or_none(value: datetime | str | None) -> str | None:
    if value is None:
        return None
    parsed = _parse_time(value if isinstance(value, str) else value.isoformat())
    assert parsed is not None
    return parsed.strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_time(value: Any) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise LifecycleError("lifecycle_invalid")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise LifecycleError("lifecycle_invalid") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


__all__ = [
    "DISTRIBUTION",
    "IMPORT_NAME",
    "INSTALL_SETS",
    "LIFECYCLE_REASON_CODES",
    "LIFECYCLE_SCHEMA_VERSION",
    "SET_ROOTS",
    "ArtifactSet",
    "LifecycleCheck",
    "LifecycleError",
    "LifecycleInputs",
    "LifecyclePlan",
    "LifecycleValidation",
    "Migration",
    "artifact_identity",
    "build_lifecycle_section",
    "is_pure_wheel",
    "normalize_distribution",
    "set_digest",
    "stage_lifecycle_inputs",
    "validate_lifecycle",
    "version_key",
]
