#!/usr/bin/env python3
"""Create deterministic, host-neutral archives of OpenMed Agent Skills.

The exporter deliberately uses only the Python standard library.  It reads
local skill files and compatibility data, so exporting a bundle never needs a
network connection or a model download.

Examples:
    python scripts/skills/export.py --output openmed-skills.zip
    python scripts/skills/export.py --pack privacy --host codex \
        --output openmed-privacy.zip --source-revision 0123456789abcdef
"""

from __future__ import annotations

import argparse
import copy
import fnmatch
import gzip
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILLS_ROOT = REPO_ROOT / "skills"
COMPATIBILITY_PATH = SKILLS_ROOT / "compatibility.json"

MANIFEST_NAME = "manifest.json"
COMPATIBILITY_NAME = "compatibility.json"
PACK_MANIFEST_NAME = "skill-packs.json"
MANIFEST_FORMAT = "openmed-agent-skill-bundle"
MANIFEST_SCHEMA_VERSION = 1
ARCHIVE_FORMATS = ("zip", "tar.gz")
COMPATIBILITY_FORMAT = "openmed-agent-skill-compatibility"
SKILL_CONTENT_DIRS = frozenset({"agents", "assets", "references", "scripts"})
SKILL_INFRASTRUCTURE_DIRS = frozenset({"packs"})
MAX_COMPATIBILITY_BYTES = 1024 * 1024
MAX_SKILL_FILE_BYTES = 25 * 1024 * 1024
MAX_BUNDLE_SOURCE_BYTES = 100 * 1024 * 1024

_IDENTIFIER_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_BUNDLE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_REVISION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$")
_PORTABLE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._-]*$")


class ExportError(ValueError):
    """Raised when a bundle cannot be created from the supplied local data."""


@dataclass(frozen=True)
class ExportResult:
    """Paths, checksums, and manifest produced by :func:`export_bundle`."""

    archive_path: Path
    manifest_path: Path
    archive_sha256: str
    manifest_sha256: str
    manifest: dict[str, Any]


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    """Serialize JSON using the stable representation used in bundles."""

    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _error(message: str) -> ExportError:
    """Build an error whose message never contains file contents."""

    return ExportError(message)


def load_compatibility(
    path: Path | os.PathLike[str] = COMPATIBILITY_PATH,
) -> dict[str, Any]:
    """Load and validate host and pack declarations from a local JSON file.

    Args:
        path: Compatibility JSON file to read.

    Returns:
        Parsed compatibility data.  The returned mapping is safe to embed in a
        manifest because it contains configuration only, not local paths or
        file contents.

    Raises:
        ExportError: If the file is missing, invalid, or incomplete.
    """

    compatibility_path = Path(path)
    if compatibility_path.is_symlink() or not compatibility_path.is_file():
        raise _error("compatibility data could not be read")
    try:
        raw = compatibility_path.read_bytes()
        if len(raw) > MAX_COMPATIBILITY_BYTES:
            raise _error("compatibility data exceeds the size limit")
        data = json.loads(raw.decode("utf-8"))
    except ExportError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        del exc
        raise _error("compatibility data could not be read") from None

    if not isinstance(data, dict):
        raise _error("compatibility data must be a JSON object")
    if data.get("format") != COMPATIBILITY_FORMAT:
        raise _error("unsupported compatibility format")
    if data.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise _error("unsupported compatibility schema version")

    hosts = data.get("hosts")
    if not isinstance(hosts, dict) or not hosts:
        raise _error("compatibility data must declare at least one host")
    for host_name, declaration in hosts.items():
        if not isinstance(host_name, str) or not _IDENTIFIER_RE.fullmatch(host_name):
            raise _error("compatibility data contains an invalid host name")
        if not isinstance(declaration, dict):
            raise _error(f"host '{host_name}' must be an object")
        skills_dir = declaration.get("skills_dir", declaration.get("skills_path"))
        if (
            not isinstance(skills_dir, str)
            or not skills_dir.strip()
            or len(skills_dir) > 512
            or any(character in skills_dir for character in "\x00\r\n")
        ):
            raise _error("a host declaration is missing a safe skills_dir")
        capabilities = declaration.get("capabilities")
        if not isinstance(capabilities, dict):
            raise _error("a host capabilities declaration must be an object")
        archive_formats = capabilities.get("archive_formats")
        if (
            not isinstance(archive_formats, list)
            or not archive_formats
            or not all(
                isinstance(value, str) and value in ARCHIVE_FORMATS
                for value in archive_formats
            )
            or len(set(archive_formats)) != len(archive_formats)
        ):
            raise _error("a host declares invalid archive formats")

    packs = data.get("packs", {})
    if not isinstance(packs, dict):
        raise _error("compatibility packs must be an object")
    for pack_name, declaration in packs.items():
        if not isinstance(pack_name, str) or not _IDENTIFIER_RE.fullmatch(pack_name):
            raise _error("compatibility data contains an invalid pack name")
        if not isinstance(declaration, (dict, list)):
            raise _error("a pack declaration must be an object or list")
        _pack_items(declaration)

    return data


def discover_skills(
    skills_root: Path | os.PathLike[str] = SKILLS_ROOT,
) -> tuple[str, ...]:
    """Return sorted skill directory names containing a regular ``SKILL.md``."""

    root = Path(skills_root)
    if root.is_symlink() or not root.is_dir():
        raise _error("skills root is not a directory")

    names: list[str] = []
    try:
        children = sorted(root.iterdir(), key=lambda item: item.name)
    except OSError:
        raise _error("skills root could not be listed") from None
    for child in children:
        if child.name.startswith((".", "_")) or child.name in SKILL_INFRASTRUCTURE_DIRS:
            continue
        skill_file = child / "SKILL.md"
        if not child.is_dir() and not child.is_symlink():
            continue
        if child.is_symlink():
            raise _error("a skill directory must not be a symlink")
        if _IDENTIFIER_RE.fullmatch(child.name) is None:
            raise _error("skills root contains an invalid skill directory")
        if skill_file.is_symlink() or not skill_file.is_file():
            raise _error("a skill directory is missing a regular SKILL.md")
        names.append(child.name)
    return tuple(names)


def _tracked_skill_files(
    skills_root: Path, skill_names: Sequence[str]
) -> set[Path] | None:
    """Return tracked selected files, or ``None`` outside a Git checkout."""

    environment = {
        key: value
        for key, value in os.environ.items()
        if key.upper()
        in {
            "COMSPEC",
            "LANG",
            "LC_ALL",
            "PATH",
            "PATHEXT",
            "SYSTEMROOT",
            "WINDIR",
        }
    }
    try:
        top_level = subprocess.run(
            ["git", "-C", str(skills_root.parent), "rev-parse", "--show-toplevel"],
            check=False,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    if top_level.returncode != 0:
        return None
    try:
        repository_root = Path(os.fsdecode(top_level.stdout.strip())).resolve()
        relative_skills_root = skills_root.resolve().relative_to(repository_root)
    except (OSError, RuntimeError, ValueError):
        return None

    pathspecs = [
        (relative_skills_root / skill_name).as_posix() for skill_name in skill_names
    ]
    try:
        tracked = subprocess.run(
            ["git", "-C", str(repository_root), "ls-files", "-z", "--", *pathspecs],
            check=False,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        raise _error("tracked skill files could not be inspected") from None
    if tracked.returncode != 0:
        raise _error("tracked skill files could not be inspected")
    return {
        (repository_root / Path(os.fsdecode(raw_path))).resolve(strict=False)
        for raw_path in tracked.stdout.split(b"\0")
        if raw_path
    }


def _pack_items(declaration: dict[str, Any] | list[Any]) -> tuple[list[str], list[str]]:
    if isinstance(declaration, list):
        if not all(isinstance(item, str) for item in declaration):
            raise _error("pack entries must be strings")
        return list(declaration), []

    skills = declaration.get("skills", [])
    patterns = declaration.get("patterns", declaration.get("include", []))
    if isinstance(skills, str):
        skills = [skills]
    if isinstance(patterns, str):
        patterns = [patterns]
    if not isinstance(skills, list) or not isinstance(patterns, list):
        raise _error("pack skills and patterns must be lists")
    if not all(isinstance(item, str) for item in skills + patterns):
        raise _error("pack entries must be strings")
    return skills, patterns


def _load_pack_manifest(
    path: Path | os.PathLike[str],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Load the canonical topical-pack manifest used by the pack builder."""

    manifest_path = Path(path)
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise _error("pack manifest could not be read")
    try:
        raw = manifest_path.read_bytes()
        if len(raw) > MAX_COMPATIBILITY_BYTES:
            raise _error("pack manifest exceeds the size limit")
        payload = json.loads(raw.decode("utf-8"))
    except ExportError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise _error("pack manifest could not be read") from None

    if not isinstance(payload, dict) or payload.get("manifest_version") != 1:
        raise _error("pack manifest has an unsupported schema")
    declarations = payload.get("packs")
    if not isinstance(declarations, list) or not declarations:
        raise _error("pack manifest must contain packs")

    packs: dict[str, dict[str, Any]] = {}
    for declaration in declarations:
        if not isinstance(declaration, dict):
            raise _error("pack manifest contains an invalid declaration")
        identifier = declaration.get("id")
        version = declaration.get("version")
        description = declaration.get("description")
        skills = declaration.get("skills")
        budget = declaration.get("budget")
        if not isinstance(identifier, str) or not _IDENTIFIER_RE.fullmatch(identifier):
            raise _error("pack manifest contains an invalid identifier")
        if identifier in packs:
            raise _error("pack manifest contains a duplicate identifier")
        if not isinstance(version, str) or not re.fullmatch(
            r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)",
            version,
        ):
            raise _error("pack manifest contains an invalid version")
        if not isinstance(description, str) or not description.strip():
            raise _error("pack manifest contains an invalid description")
        if (
            not isinstance(skills, list)
            or not skills
            or not all(
                isinstance(skill, str) and _IDENTIFIER_RE.fullmatch(skill)
                for skill in skills
            )
            or len(set(skills)) != len(skills)
        ):
            raise _error("pack manifest contains invalid skill membership")
        if not isinstance(budget, dict) or any(
            not isinstance(budget.get(key), int)
            or isinstance(budget.get(key), bool)
            or budget[key] <= 0
            for key in ("max_skills", "max_bytes")
        ):
            raise _error("pack manifest contains an invalid budget")
        packs[identifier] = {
            "budget": {
                "max_bytes": budget["max_bytes"],
                "max_skills": budget["max_skills"],
            },
            "description": description.strip(),
            "skills": sorted(skills),
            "version": version,
        }

    return payload, dict(sorted(packs.items()))


def _apply_canonical_packs(
    compatibility: Mapping[str, Any],
    canonical_packs: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    """Replace compatible duplicate declarations with canonical pack data."""

    current = compatibility.get("packs", {})
    if not isinstance(current, dict):
        raise _error("compatibility packs must be an object")
    if current:
        if set(current) != set(canonical_packs):
            raise _error("compatibility packs conflict with the canonical manifest")
        for identifier, canonical in canonical_packs.items():
            exact, patterns = _pack_items(current[identifier])
            version = (
                current[identifier].get("version")
                if isinstance(current[identifier], dict)
                else None
            )
            if (
                patterns
                or any(any(character in name for character in "*?[") for name in exact)
                or sorted(exact) != canonical["skills"]
                or (version is not None and version != canonical["version"])
            ):
                raise _error("compatibility packs conflict with the canonical manifest")

    merged = copy.deepcopy(dict(compatibility))
    merged["packs"] = copy.deepcopy(dict(canonical_packs))
    return merged


def select_skills(
    available: Sequence[str],
    compatibility: Mapping[str, Any],
    skill_names: Iterable[str] = (),
    pack_names: Iterable[str] = (),
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Resolve explicit skills and data-defined packs to sorted skill names.

    An empty selection means all discovered skills.  Pack declarations may
    contain exact ``skills`` entries or glob-style ``patterns`` entries, which
    keeps pack membership data-driven as the catalog grows.

    Args:
        available: Skill names discovered below the local skills root.
        compatibility: Parsed compatibility data.
        skill_names: Exact skill names requested by the caller.
        pack_names: Pack names declared in ``compatibility``.

    Returns:
        A pair of ``(selected_skills, selected_packs)``.
    """

    available_set = set(available)
    requested_skills = tuple(skill_names)
    requested_packs = tuple(pack_names)
    if not requested_skills and not requested_packs:
        return tuple(sorted(available_set)), ()

    selected: set[str] = set()
    for name in requested_skills:
        if name not in available_set:
            raise _error("skill selection contains an unknown identifier")
        selected.add(name)

    packs = compatibility.get("packs", {})
    for pack_name in requested_packs:
        declaration = packs.get(pack_name) if isinstance(packs, dict) else None
        if declaration is None:
            raise _error("pack selection contains an unknown identifier")
        exact, patterns = _pack_items(declaration)
        for name in exact:
            if any(char in name for char in "*?["):
                patterns.append(name)
            elif name not in available_set:
                raise _error("a pack references an unknown skill")
            else:
                selected.add(name)
        for pattern in patterns:
            matches = {
                name for name in available_set if fnmatch.fnmatchcase(name, pattern)
            }
            if not matches:
                raise _error("a pack pattern selects no available skills")
            selected.update(matches)
        if not exact and not patterns:
            raise _error("a pack selects no skills")

    if not selected:
        raise _error("selection contains no skills")
    return tuple(sorted(selected)), tuple(sorted(set(requested_packs)))


def _collect_skill_files(
    skills_root: Path,
    skill_names: Sequence[str],
    *,
    tracked_files: set[Path] | None = None,
) -> list[tuple[str, bytes]]:
    """Read bounded, portable skill files without following symlinks."""

    members: list[tuple[str, bytes]] = []
    portable_names: set[str] = set()
    total_size = 0
    for skill_name in skill_names:
        skill_dir = skills_root / skill_name
        if not skill_dir.is_dir() or skill_dir.is_symlink():
            raise _error("a selected skill is not a regular directory")
        try:
            paths = sorted(skill_dir.rglob("*"), key=lambda item: item.as_posix())
        except OSError:
            raise _error("a selected skill could not be listed") from None
        for path in paths:
            if path.is_symlink():
                raise _error("a selected skill contains an unsupported symlink")
            if path.is_dir():
                continue
            if not path.is_file():
                raise _error("a selected skill contains an unsupported file")
            if tracked_files is not None and path.resolve() not in tracked_files:
                raise _error("a selected skill contains an untracked file")

            relative_to_skill = path.relative_to(skill_dir)
            parts = relative_to_skill.parts
            if not parts:
                raise _error("a selected skill contains an invalid path")
            if len(parts) == 1:
                if parts[0] != "SKILL.md":
                    raise _error("a selected skill contains an unsupported root file")
            elif parts[0] not in SKILL_CONTENT_DIRS:
                raise _error("a selected skill contains an unsupported directory")
            if any(
                component.startswith(".")
                or _PORTABLE_COMPONENT_RE.fullmatch(component) is None
                for component in parts
            ):
                raise _error("a selected skill contains a non-portable path")

            try:
                size = path.stat().st_size
            except OSError:
                raise _error("a selected skill could not be inspected") from None
            if size > MAX_SKILL_FILE_BYTES:
                raise _error("a selected skill file exceeds the size limit")
            total_size += size
            if total_size > MAX_BUNDLE_SOURCE_BYTES:
                raise _error("selected skill files exceed the bundle size limit")

            relative = path.relative_to(skills_root).as_posix()
            member_name = f"skills/{relative}"
            portable_name = unicodedata.normalize("NFC", member_name).casefold()
            if portable_name in portable_names:
                raise _error("selected skills contain a portable path collision")
            portable_names.add(portable_name)
            try:
                content = path.read_bytes()
            except OSError:
                raise _error("a selected skill could not be read") from None
            if len(content) != size:
                raise _error("a selected skill changed during export")
            members.append((member_name, content))
    return members


def resolve_source_revision(repo_root: Path | os.PathLike[str] = REPO_ROOT) -> str:
    """Return the local Git revision, or ``unknown`` when no Git metadata exists.

    The command is strictly local: it is ``git rev-parse`` only and never
    contacts a remote.  A bundle remains usable when exported from a source
    archive without Git metadata.
    """

    environment = {
        key: value
        for key, value in os.environ.items()
        if key.upper()
        in {
            "COMSPEC",
            "LANG",
            "LC_ALL",
            "PATH",
            "PATHEXT",
            "SYSTEMROOT",
            "WINDIR",
        }
    }
    try:
        completed = subprocess.run(
            ["git", "-C", str(Path(repo_root)), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            env=environment,
            stdin=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return "unknown"
    revision = completed.stdout.strip()
    return revision if _REVISION_RE.fullmatch(revision) else "unknown"


def _validate_revision(revision: str) -> str:
    if not isinstance(revision, str) or not _REVISION_RE.fullmatch(revision):
        raise _error("source revision must be a non-empty safe identifier")
    return revision


def _select_hosts(
    compatibility: Mapping[str, Any], host_names: Iterable[str] | None
) -> dict[str, Any]:
    hosts = compatibility["hosts"]
    selected_names = sorted(hosts) if host_names is None else sorted(set(host_names))
    selected: dict[str, Any] = {}
    for host_name in selected_names:
        if host_name not in hosts:
            raise _error("host selection contains an unknown identifier")
        selected[host_name] = copy.deepcopy(hosts[host_name])
    if not selected:
        raise _error("selection contains no hosts")
    return selected


def _validate_host_archive_format(
    hosts: Mapping[str, Any], archive_format: str
) -> None:
    """Require every selected host to declare support for the output format."""

    for declaration in hosts.values():
        capabilities = declaration["capabilities"]
        if archive_format not in capabilities["archive_formats"]:
            raise _error("a selected host does not support the archive format")


def _normalise_archive_format(archive_format: str | None, output: Path) -> str:
    value = archive_format.lower().lstrip(".") if archive_format else ""
    lower_name = output.name.lower()
    inferred: str | None = None
    if lower_name.endswith((".tar.gz", ".tgz")):
        inferred = "tar.gz"
    elif lower_name.endswith(".zip"):
        inferred = "zip"
    if not value:
        value = inferred or "zip"
    if value in {"tgz", "tar-gz"}:
        value = "tar.gz"
    if value not in ARCHIVE_FORMATS:
        raise _error("archive format must be zip or tar.gz")
    if inferred is not None and inferred != value:
        raise _error("archive format conflicts with the output filename")
    return value


def _default_manifest_path(output: Path) -> Path:
    lower_name = output.name.lower()
    if lower_name.endswith(".tar.gz"):
        stem = output.name[: -len(".tar.gz")]
    elif lower_name.endswith(".tgz"):
        stem = output.name[: -len(".tgz")]
    elif output.suffix:
        stem = output.name[: -len(output.suffix)]
    else:
        stem = output.name
    return output.with_name(f"{stem}.manifest.json")


def _file_records(members: Sequence[tuple[str, bytes]]) -> list[dict[str, Any]]:
    records = [
        {"path": path, "sha256": _sha256(content), "size": len(content)}
        for path, content in members
    ]
    return sorted(records, key=lambda record: record["path"])


def build_manifest(
    *,
    bundle_name: str,
    skill_names: Sequence[str],
    pack_names: Sequence[str],
    hosts: Mapping[str, Any],
    members: Sequence[tuple[str, bytes]],
    source_revision: str,
    archive_format: str,
) -> dict[str, Any]:
    """Build the stable manifest embedded in and written beside a bundle."""

    records = _file_records(members)
    checksums = {record["path"]: record["sha256"] for record in records}
    return {
        "archive": {
            "format": archive_format,
            "manifest_path": MANIFEST_NAME,
            "skills_root": "skills",
        },
        "checksums": {"algorithm": "sha256", "files": checksums},
        "files": records,
        "format": MANIFEST_FORMAT,
        "bundle_name": bundle_name,
        "hosts": copy.deepcopy(dict(hosts)),
        "packs": list(pack_names),
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "skills": list(skill_names),
        "source": {"directory": "skills", "revision": source_revision},
        "source_revision": source_revision,
    }


def _write_zip(path: Path, members: Sequence[tuple[str, bytes]]) -> None:
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for name, content in members:
            info = zipfile.ZipInfo(filename=name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_STORED
            info.extra = b""
            info.comment = b""
            archive.writestr(info, content)


def _write_tar_gz(path: Path, members: Sequence[tuple[str, bytes]]) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed:
            with tarfile.open(
                fileobj=compressed, mode="w", format=tarfile.USTAR_FORMAT
            ) as archive:
                for name, content in members:
                    info = tarfile.TarInfo(name=name)
                    info.size = len(content)
                    info.mtime = 0
                    info.mode = 0o644
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    archive.addfile(info, io.BytesIO(content))


def _write_archive(
    path: Path, members: Sequence[tuple[str, bytes]], archive_format: str
) -> None:
    try:
        if archive_format == "zip":
            _write_zip(path, members)
        else:
            _write_tar_gz(path, members)
    except (OSError, OverflowError, ValueError, tarfile.TarError, zipfile.BadZipFile):
        raise _error("bundle archive could not be written") from None


def _temporary_path(target: Path) -> Path:
    try:
        descriptor, name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
        )
    except OSError:
        raise _error("bundle output directory is not available") from None
    try:
        os.close(descriptor)
    except OSError:
        try:
            Path(name).unlink(missing_ok=True)
        except OSError:
            pass
        raise _error("bundle output directory is not available") from None
    return Path(name)


def _validate_output_locations(
    output: Path,
    sidecar: Path,
    *,
    skills_root: Path,
    compatibility_path: Path,
    force: bool,
) -> None:
    """Reject aliases, source-tree outputs, symlinks, and unsafe overwrites."""

    try:
        canonical_output = output.resolve(strict=False)
        canonical_sidecar = sidecar.resolve(strict=False)
        canonical_skills = skills_root.resolve(strict=False)
        canonical_compatibility = compatibility_path.resolve(strict=False)
    except (OSError, RuntimeError):
        raise _error("bundle output locations could not be resolved") from None
    if canonical_output == canonical_sidecar:
        raise _error("archive and manifest paths must be different")
    for target in (canonical_output, canonical_sidecar):
        if target.is_relative_to(canonical_skills):
            raise _error("bundle outputs must be outside the skills source tree")
        if target == canonical_compatibility:
            raise _error("bundle outputs must not replace compatibility data")

    for target in (output, sidecar):
        if target.is_symlink():
            raise _error("bundle output targets must not be symlinks")
        if target.exists() and not target.is_file():
            raise _error("bundle output targets must be regular files")
        if target.exists() and not force:
            raise _error("refusing to overwrite an existing bundle; pass force=True")


def _rollback_outputs(
    prepared: Sequence[Path], backups: Mapping[Path, Path | None]
) -> bool:
    """Best-effort rollback of prepared output targets."""

    ok = True
    for target in reversed(prepared):
        try:
            target.unlink(missing_ok=True)
        except OSError:
            ok = False
        backup = backups.get(target)
        if backup is not None and backup.exists():
            try:
                os.replace(backup, target)
            except OSError:
                ok = False
    return ok


def _finalize_outputs(
    archive_temp: Path,
    output: Path,
    manifest_temp: Path,
    sidecar: Path,
    *,
    force: bool,
) -> None:
    """Finalize both staged files, restoring existing files after any failure."""

    staged = ((archive_temp, output), (manifest_temp, sidecar))
    prepared: list[Path] = []
    backups: dict[Path, Path | None] = {}
    try:
        for _, target in staged:
            backup: Path | None = None
            if target.is_symlink() or (target.exists() and not target.is_file()):
                raise _error("bundle output targets changed during export")
            if target.exists():
                if not force:
                    raise _error("bundle output targets changed during export")
                backup = _temporary_path(target)
                backup.unlink()
                os.replace(target, backup)
            backups[target] = backup
            prepared.append(target)
            descriptor = os.open(
                target,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
            os.close(descriptor)

        for temporary, target in staged:
            os.replace(temporary, target)
    except (OSError, ExportError):
        rolled_back = _rollback_outputs(prepared, backups)
        if not rolled_back:
            raise _error(
                "bundle outputs could not be finalized or rolled back"
            ) from None
        raise _error("bundle outputs could not be finalized") from None
    else:
        for backup in backups.values():
            if backup is not None:
                try:
                    backup.unlink(missing_ok=True)
                except OSError:
                    pass


def export_bundle(
    output: Path | os.PathLike[str],
    *,
    skills_root: Path | os.PathLike[str] = SKILLS_ROOT,
    skills: Iterable[str] = (),
    packs: Iterable[str] = (),
    hosts: Iterable[str] | None = None,
    compatibility_path: Path | os.PathLike[str] = COMPATIBILITY_PATH,
    pack_manifest_path: Path | os.PathLike[str] | None = None,
    source_revision: str | None = None,
    archive_format: str | None = None,
    manifest_path: Path | os.PathLike[str] | None = None,
    bundle_name: str = "openmed-skills",
    force: bool = False,
) -> ExportResult:
    """Export selected local skills to a deterministic archive and manifest.

    Args:
        output: Archive path to create.
        skills_root: Directory containing skill folders.
        skills: Exact skill names to include.  Empty means all skills unless a
            pack is selected.
        packs: Pack names from the compatibility JSON.
        hosts: Host IDs to describe in the manifest.  ``None`` includes every
            declared host.
        compatibility_path: Local host/pack declaration file.
        pack_manifest_path: Optional canonical topical-pack manifest. By
            default, ``skills_root/packs/manifest.json`` is used when present.
            Its versioned declarations replace matching compatibility
            declarations and are embedded in the bundle.
        source_revision: Safe source identifier.  When omitted, the local Git
            ``HEAD`` is resolved without contacting a remote.
        archive_format: ``zip`` or ``tar.gz``.  By default it is inferred from
            the output suffix, falling back to ``zip``.
        manifest_path: Optional sidecar manifest path.
        bundle_name: Stable human-readable bundle identifier.
        force: Replace existing regular archive and manifest files, restoring
            both prior outputs if finalization fails.

    Returns:
        An :class:`ExportResult` containing both output checksums.

    Raises:
        ExportError: If selection, source data, or output safety checks fail.
    """

    output_path = Path(output)
    sidecar_path = (
        Path(manifest_path) if manifest_path else _default_manifest_path(output_path)
    )
    if not bundle_name or not _BUNDLE_NAME_RE.fullmatch(bundle_name):
        raise _error("bundle name must be a non-empty safe identifier")

    skills_root_path = Path(skills_root)
    compatibility_file = Path(compatibility_path)
    _validate_output_locations(
        output_path,
        sidecar_path,
        skills_root=skills_root_path,
        compatibility_path=compatibility_file,
        force=force,
    )
    compatibility = load_compatibility(compatibility_file)
    pack_manifest_bytes: bytes | None = None
    pack_manifest_file = (
        Path(pack_manifest_path)
        if pack_manifest_path is not None
        else skills_root_path / "packs" / "manifest.json"
    )
    if pack_manifest_file.is_symlink() or pack_manifest_file.exists():
        pack_payload, canonical_packs = _load_pack_manifest(pack_manifest_file)
        compatibility = _apply_canonical_packs(compatibility, canonical_packs)
        pack_manifest_bytes = _canonical_json(pack_payload)
    available = discover_skills(skills_root_path)
    selected_skills, selected_packs = select_skills(
        available,
        compatibility,
        skill_names=skills,
        pack_names=packs,
    )
    selected_hosts = _select_hosts(compatibility, hosts)
    revision = _validate_revision(
        resolve_source_revision(skills_root_path.parent)
        if source_revision is None
        else source_revision
    )
    archive_kind = _normalise_archive_format(archive_format, output_path)
    if hosts is not None:
        _validate_host_archive_format(selected_hosts, archive_kind)

    tracked_files = _tracked_skill_files(skills_root_path, selected_skills)
    skill_members = _collect_skill_files(
        skills_root_path,
        selected_skills,
        tracked_files=tracked_files,
    )
    compatibility_bytes = _canonical_json(compatibility)
    source_members = [(COMPATIBILITY_NAME, compatibility_bytes)]
    if pack_manifest_bytes is not None:
        source_members.append((PACK_MANIFEST_NAME, pack_manifest_bytes))
    source_members.extend(skill_members)
    manifest = build_manifest(
        bundle_name=bundle_name,
        skill_names=selected_skills,
        pack_names=selected_packs,
        hosts=selected_hosts,
        members=source_members,
        source_revision=revision,
        archive_format=archive_kind,
    )
    manifest_bytes = _canonical_json(manifest)
    archive_members = [
        (MANIFEST_NAME, manifest_bytes),
        *sorted(source_members, key=lambda item: item[0]),
    ]

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        raise _error("bundle output directory is not available") from None
    _validate_output_locations(
        output_path,
        sidecar_path,
        skills_root=skills_root_path,
        compatibility_path=compatibility_file,
        force=force,
    )

    archive_temp: Path | None = None
    manifest_temp: Path | None = None
    archive_sha256: str | None = None
    try:
        archive_temp = _temporary_path(output_path)
        manifest_temp = _temporary_path(sidecar_path)
        _write_archive(archive_temp, archive_members, archive_kind)
        try:
            archive_sha256 = _sha256_file(archive_temp)
        except OSError:
            raise _error("bundle archive could not be read after writing") from None
        try:
            manifest_temp.write_bytes(manifest_bytes)
        except OSError:
            raise _error("bundle manifest could not be written") from None
        _finalize_outputs(
            archive_temp,
            output_path,
            manifest_temp,
            sidecar_path,
            force=force,
        )
        archive_temp = None
        manifest_temp = None
    finally:
        for temporary in (archive_temp, manifest_temp):
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass

    assert archive_sha256 is not None
    return ExportResult(
        archive_path=output_path,
        manifest_path=sidecar_path,
        archive_sha256=archive_sha256,
        manifest_sha256=_sha256(manifest_bytes),
        manifest=manifest,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("skill_names", nargs="*", metavar="SKILL")
    parser.add_argument("--output", required=True, type=Path, help="archive path")
    parser.add_argument("--manifest", type=Path, help="sidecar manifest path")
    parser.add_argument("--skills-root", type=Path, default=SKILLS_ROOT)
    parser.add_argument("--compatibility", type=Path, default=COMPATIBILITY_PATH)
    parser.add_argument("--pack-manifest", type=Path)
    parser.add_argument("--skill", action="append", dest="selected_skills")
    parser.add_argument("--pack", action="append", dest="selected_packs")
    parser.add_argument("--host", action="append", dest="selected_hosts")
    parser.add_argument("--source-revision")
    parser.add_argument("--format", dest="archive_format", choices=ARCHIVE_FORMATS)
    parser.add_argument("--name", default="openmed-skills", dest="bundle_name")
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace existing archive and manifest outputs",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line exporter and return a process status code."""

    args = _parser().parse_args(argv)
    selected_skills = [*(args.selected_skills or []), *args.skill_names]
    try:
        result = export_bundle(
            args.output,
            skills_root=args.skills_root,
            skills=selected_skills,
            packs=args.selected_packs or [],
            hosts=args.selected_hosts,
            compatibility_path=args.compatibility,
            pack_manifest_path=args.pack_manifest,
            source_revision=args.source_revision,
            archive_format=args.archive_format,
            manifest_path=args.manifest,
            bundle_name=args.bundle_name,
            force=args.force,
        )
    except ExportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Archive: {result.archive_path}")
    print(f"Manifest: {result.manifest_path}")
    print(f"Archive SHA-256: {result.archive_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
