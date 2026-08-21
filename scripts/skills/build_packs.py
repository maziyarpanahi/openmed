#!/usr/bin/env python3
"""Build deterministic, offline Agent Skill packs from a versioned manifest.

The generated packs contain metadata and relative symlinks to the source skill
directories.  Skill content is therefore kept in one place and is never
copied into a pack.  Use ``--selection-only`` when symlinks are unavailable.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "skills" / "packs" / "manifest.json"
DEFAULT_SKILLS_DIR = REPO_ROOT / "skills"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "build" / "skill-packs"
MANIFEST_VERSION = 1
IDENTIFIER_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
SEMVER_RE = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")


class PackValidationError(ValueError):
    """Raised when the manifest or skill source tree is invalid."""

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors = tuple(errors)
        super().__init__("pack manifest validation failed")


class PackBuildError(RuntimeError):
    """Raised when a generated pack cannot be written safely."""


@dataclass(frozen=True)
class PackBudget:
    """Limits applied to one topical pack."""

    max_skills: int
    max_bytes: int


@dataclass(frozen=True)
class PackSpec:
    """One versioned topical pack declaration."""

    identifier: str
    version: str
    description: str
    skills: tuple[str, ...]
    budget: PackBudget


@dataclass(frozen=True)
class PackManifest:
    """Parsed pack manifest."""

    manifest_version: int
    packs: tuple[PackSpec, ...]


@dataclass(frozen=True)
class PackReport:
    """Validated size information for one pack."""

    pack: PackSpec
    skill_count: int
    size_bytes: int


def _read_json(path: Path) -> object:
    """Read JSON without echoing source content in errors."""

    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        raise PackValidationError(("unable to read the pack manifest",)) from None

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise PackValidationError(
            (
                f"pack manifest is not valid JSON at line {exc.lineno}, column {exc.colno}",
            )
        ) from None


def _valid_identifier(value: object) -> bool:
    text = _plain_text(value)
    return text is not None and bool(IDENTIFIER_RE.fullmatch(text))


def _valid_version(value: object) -> bool:
    text = _plain_text(value)
    return text is not None and bool(SEMVER_RE.fullmatch(text))


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _plain_text(value: object) -> str | None:
    """Copy a string into a base ``str`` without calling subclass hooks."""

    if not isinstance(value, str):
        return None
    try:
        return str.encode(value, "utf-8").decode("utf-8")
    except Exception:
        return None


def _canonical_manifest(manifest: PackManifest) -> PackManifest:
    """Validate and canonicalize an in-memory manifest before filesystem use."""

    if not isinstance(manifest, PackManifest):
        raise PackValidationError(("pack manifest is invalid",))
    if (
        type(manifest.manifest_version) is not int
        or manifest.manifest_version != MANIFEST_VERSION
    ):
        raise PackValidationError(("manifest_version must be the supported version",))
    try:
        raw_packs = tuple(manifest.packs)
    except Exception:
        raise PackValidationError(("packs could not be consumed safely",)) from None
    if not raw_packs:
        raise PackValidationError(("packs must be a non-empty list",))

    errors: list[str] = []
    canonical: list[PackSpec] = []
    seen_pack_ids: set[str] = set()
    membership: dict[str, str] = {}
    for index, pack in enumerate(raw_packs):
        prefix = f"packs[{index}]"
        if not isinstance(pack, PackSpec) or not isinstance(pack.budget, PackBudget):
            errors.append(f"{prefix} is invalid")
            continue
        identifier = _plain_text(pack.identifier)
        version = _plain_text(pack.version)
        description = _plain_text(pack.description)
        if identifier is None or not _valid_identifier(identifier):
            errors.append(f"{prefix}.id is invalid")
            continue
        if identifier in seen_pack_ids:
            errors.append(f"pack id '{identifier}' is duplicated")
            continue
        seen_pack_ids.add(identifier)
        if version is None or not _valid_version(version):
            errors.append(f"{prefix}.version is invalid")
            continue
        if description is None or not description.strip():
            errors.append(f"{prefix}.description is invalid")
            continue
        if not _positive_int(pack.budget.max_skills) or not _positive_int(
            pack.budget.max_bytes
        ):
            errors.append(f"{prefix}.budget is invalid")
            continue
        try:
            raw_skills = tuple(pack.skills)
        except Exception:
            errors.append(f"{prefix}.skills could not be consumed safely")
            continue
        if not raw_skills:
            errors.append(f"{prefix}.skills must be a non-empty list")
            continue

        skills: list[str] = []
        seen_skills: set[str] = set()
        for skill_index, raw_skill in enumerate(raw_skills):
            skill_id = _plain_text(raw_skill)
            if skill_id is None or not _valid_identifier(skill_id):
                errors.append(f"{prefix}.skills[{skill_index}] is invalid")
                continue
            if skill_id in seen_skills:
                errors.append(f"{prefix}.skills lists '{skill_id}' more than once")
                continue
            seen_skills.add(skill_id)
            previous = membership.get(skill_id)
            if previous is not None and previous != identifier:
                errors.append(
                    f"skill '{skill_id}' is assigned to both '{previous}' and "
                    f"'{identifier}'"
                )
            else:
                membership[skill_id] = identifier
            skills.append(skill_id)

        canonical.append(
            PackSpec(
                identifier=identifier,
                version=version,
                description=description.strip(),
                skills=tuple(sorted(skills)),
                budget=PackBudget(
                    max_skills=pack.budget.max_skills,
                    max_bytes=pack.budget.max_bytes,
                ),
            )
        )

    if errors:
        raise PackValidationError(tuple(errors))
    return PackManifest(
        manifest_version=MANIFEST_VERSION,
        packs=tuple(sorted(canonical, key=lambda pack: pack.identifier)),
    )


def load_manifest(path: Path = DEFAULT_MANIFEST) -> PackManifest:
    """Load and validate the manifest schema from ``path``.

    Source-tree checks such as missing skill folders and byte budgets are
    performed by :func:`validate_manifest`, which also receives the skills
    directory to inspect.
    """

    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise PackValidationError(("pack manifest must be a JSON object",))

    errors: list[str] = []
    raw_manifest_version = payload.get("manifest_version")
    if (
        type(raw_manifest_version) is not int
        or raw_manifest_version != MANIFEST_VERSION
    ):
        errors.append("manifest_version must be the supported version")

    raw_packs = payload.get("packs")
    if not isinstance(raw_packs, list) or not raw_packs:
        errors.append("packs must be a non-empty list")
        raise PackValidationError(errors)

    packs: list[PackSpec] = []
    seen_pack_ids: set[str] = set()

    for index, raw_pack in enumerate(raw_packs):
        prefix = f"packs[{index}]"
        if not isinstance(raw_pack, dict):
            errors.append(f"{prefix} must be an object")
            continue

        pack_errors: list[str] = []
        identifier = raw_pack.get("id")
        if not _valid_identifier(identifier):
            pack_errors.append(f"{prefix}.id is invalid")

        version = raw_pack.get("version")
        if not _valid_version(version):
            pack_errors.append(f"{prefix}.version is invalid")

        description = raw_pack.get("description")
        if not isinstance(description, str) or not description.strip():
            pack_errors.append(f"{prefix}.description is invalid")

        raw_skills = raw_pack.get("skills")
        skill_ids: list[str] = []
        if not isinstance(raw_skills, list) or not raw_skills:
            pack_errors.append(f"{prefix}.skills must be a non-empty list")
        else:
            for skill_index, skill_id in enumerate(raw_skills):
                if not _valid_identifier(skill_id):
                    pack_errors.append(f"{prefix}.skills[{skill_index}] is invalid")
                    continue
                if skill_id in skill_ids:
                    pack_errors.append(
                        f"{prefix}.skills lists '{skill_id}' more than once"
                    )
                    continue
                skill_ids.append(skill_id)

        raw_budget = raw_pack.get("budget")
        max_skills: object = None
        max_bytes: object = None
        if not isinstance(raw_budget, dict):
            pack_errors.append(f"{prefix}.budget must be an object")
        else:
            max_skills = raw_budget.get("max_skills")
            max_bytes = raw_budget.get("max_bytes")
            if not _positive_int(max_skills):
                pack_errors.append(f"{prefix}.budget.max_skills is invalid")
            if not _positive_int(max_bytes):
                pack_errors.append(f"{prefix}.budget.max_bytes is invalid")

        if pack_errors:
            errors.extend(pack_errors)
            continue

        assert isinstance(identifier, str)
        assert isinstance(version, str)
        assert isinstance(description, str)
        assert isinstance(max_skills, int) and not isinstance(max_skills, bool)
        assert isinstance(max_bytes, int) and not isinstance(max_bytes, bool)

        if identifier in seen_pack_ids:
            errors.append(f"pack id '{identifier}' is duplicated")
            continue
        seen_pack_ids.add(identifier)
        packs.append(
            PackSpec(
                identifier=identifier,
                version=version,
                description=description.strip(),
                skills=tuple(sorted(skill_ids)),
                budget=PackBudget(max_skills=max_skills, max_bytes=max_bytes),
            )
        )

    membership: dict[str, str] = {}
    for pack in packs:
        for skill_id in pack.skills:
            previous = membership.get(skill_id)
            if previous is not None:
                errors.append(
                    f"skill '{skill_id}' is assigned to both '{previous}' and "
                    f"'{pack.identifier}'"
                )
            else:
                membership[skill_id] = pack.identifier

    if errors:
        raise PackValidationError(tuple(errors))

    return _canonical_manifest(
        PackManifest(
            manifest_version=MANIFEST_VERSION,
            packs=tuple(sorted(packs, key=lambda pack: pack.identifier)),
        )
    )


def discover_skills(skills_dir: Path = DEFAULT_SKILLS_DIR) -> dict[str, Path]:
    """Return stable skill identifiers and their source directories."""

    if skills_dir.is_symlink() or not skills_dir.is_dir():
        raise PackValidationError(("skills directory is missing or invalid",))

    skills: dict[str, Path] = {}
    try:
        entries = sorted(skills_dir.iterdir(), key=lambda path: path.name)
    except OSError:
        raise PackValidationError(("unable to inspect the skills directory",)) from None

    for entry in entries:
        if entry.name.startswith((".", "_")):
            continue
        if entry.is_symlink() or not entry.is_dir():
            continue
        skill_file = entry / "SKILL.md"
        if skill_file.is_file() and not skill_file.is_symlink():
            skills[entry.name] = entry
    return skills


def skill_size_bytes(skill_dir: Path) -> int:
    """Return the regular-file byte size of one skill without following links."""

    def walk_error(_error: OSError) -> None:
        raise PackValidationError(("unable to measure a skill directory",))

    total = 0
    for root, directories, filenames in os.walk(
        skill_dir,
        followlinks=False,
        onerror=walk_error,
    ):
        root_path = Path(root)
        directories[:] = sorted(
            name for name in directories if not (root_path / name).is_symlink()
        )
        for filename in sorted(filenames):
            file_path = root_path / filename
            if file_path.is_symlink():
                continue
            try:
                file_stat = file_path.stat(follow_symlinks=False)
            except OSError:
                raise PackValidationError(
                    ("unable to measure a skill directory",)
                ) from None
            if stat.S_ISREG(file_stat.st_mode):
                total += file_stat.st_size
    return total


def validate_manifest(
    manifest: PackManifest, skills_dir: Path = DEFAULT_SKILLS_DIR
) -> tuple[PackReport, ...]:
    """Validate skill membership, duplicate assignments, and pack budgets."""

    manifest = _canonical_manifest(manifest)
    skills = discover_skills(skills_dir)
    sizes = {skill_id: skill_size_bytes(path) for skill_id, path in skills.items()}
    errors: list[str] = []
    membership: dict[str, str] = {}
    reports: list[PackReport] = []

    for pack in manifest.packs:
        seen_in_pack: set[str] = set()
        missing: list[str] = []
        size_bytes = 0
        for skill_id in pack.skills:
            if skill_id in seen_in_pack:
                errors.append(
                    f"pack '{pack.identifier}' lists '{skill_id}' more than once"
                )
            seen_in_pack.add(skill_id)

            previous = membership.get(skill_id)
            if previous is not None and previous != pack.identifier:
                errors.append(
                    f"skill '{skill_id}' is assigned to both '{previous}' and "
                    f"'{pack.identifier}'"
                )
            else:
                membership[skill_id] = pack.identifier

            source = skills.get(skill_id)
            if source is None:
                missing.append(skill_id)
            else:
                size_bytes += sizes[skill_id]

        if missing:
            errors.append(
                f"pack '{pack.identifier}' references missing skill(s): "
                + ", ".join(sorted(missing))
            )
        if len(pack.skills) > pack.budget.max_skills:
            errors.append(f"pack '{pack.identifier}' exceeds its skill-count budget")
        if size_bytes > pack.budget.max_bytes:
            errors.append(f"pack '{pack.identifier}' exceeds its byte budget")

        reports.append(
            PackReport(
                pack=pack,
                skill_count=len(pack.skills),
                size_bytes=size_bytes,
            )
        )

    if errors:
        raise PackValidationError(tuple(errors))
    return tuple(reports)


def _ensure_directory(path: Path, label: str) -> None:
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        raise PackBuildError(f"{label} is not a safe directory")
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        raise PackBuildError(f"could not create {label}") from None


def _resolved_path(path: Path, label: str) -> Path:
    """Resolve ``path`` for containment checks without exposing it in errors."""

    try:
        return path.resolve(strict=False)
    except OSError:
        raise PackBuildError(f"{label} is not a safe directory") from None


def _expected_link_target(source: Path, destination: Path) -> str:
    return os.path.relpath(source, destination.parent)


def _validate_existing_skill_link(
    source: Path, destination: Path, pack_id: str
) -> None:
    expected_target = _expected_link_target(source, destination)
    try:
        actual_target = os.readlink(destination)
        resolves_to_source = destination.resolve(strict=False) == source.resolve()
    except OSError:
        raise PackBuildError(
            f"pack '{pack_id}' contains an unsafe skill link"
        ) from None
    if actual_target != expected_target or not resolves_to_source:
        raise PackBuildError(f"pack '{pack_id}' contains an unexpected skill link")


def _validate_existing_metadata(metadata_path: Path, pack_id: str) -> None:
    if metadata_path.is_symlink() or (
        metadata_path.exists() and not metadata_path.is_file()
    ):
        raise PackBuildError(f"pack '{pack_id}' metadata is not safe")
    if not metadata_path.exists():
        return
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise PackBuildError(f"pack '{pack_id}' metadata is not safe") from None
    if not isinstance(payload, dict) or payload.get("pack_id") != pack_id:
        raise PackBuildError(f"pack '{pack_id}' metadata is not owned by this pack")


def _preflight_pack_directory(
    pack_dir: Path,
    report: PackReport,
    skills: dict[str, Path],
    *,
    selection_only: bool,
) -> None:
    """Reject foreign or stale output before any pack metadata is rewritten."""

    pack_id = report.pack.identifier
    if pack_dir.is_symlink() or (pack_dir.exists() and not pack_dir.is_dir()):
        raise PackBuildError(f"pack '{pack_id}' is not a safe directory")
    if not pack_dir.exists():
        return

    try:
        entries = {entry.name: entry for entry in pack_dir.iterdir()}
    except OSError:
        raise PackBuildError(f"could not inspect pack '{pack_id}'") from None

    allowed_entries = {"pack.json"}
    if not selection_only:
        allowed_entries.add("skills")
    if set(entries) - allowed_entries:
        raise PackBuildError(f"pack '{pack_id}' contains unexpected output")

    _validate_existing_metadata(pack_dir / "pack.json", pack_id)
    links_dir = pack_dir / "skills"
    if selection_only:
        return
    if links_dir.is_symlink():
        raise PackBuildError(f"pack '{pack_id}' skills are not a safe directory")
    if not links_dir.exists():
        return
    if not links_dir.is_dir():
        raise PackBuildError(f"pack '{pack_id}' skills are not a safe directory")

    expected_skills = set(report.pack.skills)
    try:
        links = tuple(links_dir.iterdir())
    except OSError:
        raise PackBuildError(f"could not inspect pack '{pack_id}' skills") from None
    for destination in links:
        if destination.name not in expected_skills or not destination.is_symlink():
            raise PackBuildError(f"pack '{pack_id}' contains stale skill output")
        _validate_existing_skill_link(skills[destination.name], destination, pack_id)


def _write_pack_metadata(
    output_dir: Path, manifest: PackManifest, report: PackReport
) -> None:
    pack_dir = output_dir / report.pack.identifier
    metadata = {
        "budget": {
            "max_bytes": report.pack.budget.max_bytes,
            "max_skills": report.pack.budget.max_skills,
        },
        "description": report.pack.description,
        "manifest_version": manifest.manifest_version,
        "pack_id": report.pack.identifier,
        "pack_version": report.pack.version,
        "size_bytes": report.size_bytes,
        "skills": list(report.pack.skills),
    }
    metadata_path = pack_dir / "pack.json"
    if metadata_path.is_symlink() or (
        metadata_path.exists() and not metadata_path.is_file()
    ):
        raise PackBuildError(f"pack '{report.pack.identifier}' metadata is not safe")
    payload = (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary_path: Path | None = None
    descriptor: int | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".pack-json-",
            suffix=".tmp",
            dir=os.fspath(pack_dir),
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _validate_existing_metadata(metadata_path, report.pack.identifier)
        os.replace(os.fspath(temporary_path), os.fspath(metadata_path))
        temporary_path = None
    except PackBuildError:
        raise
    except OSError:
        raise PackBuildError(
            f"could not write pack '{report.pack.identifier}'"
        ) from None
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except OSError:
                pass


def _link_skill(source: Path, destination: Path, pack_id: str) -> None:
    if destination.is_symlink():
        _validate_existing_skill_link(source, destination, pack_id)
        return
    if destination.exists():
        raise PackBuildError(f"pack '{pack_id}' would overwrite an existing skill")

    target = _expected_link_target(source, destination)
    try:
        destination.symlink_to(target, target_is_directory=True)
    except OSError:
        raise PackBuildError(
            f"could not link skills for pack '{pack_id}'; use --selection-only "
            "when directory symlinks are unavailable"
        ) from None


def build_packs(
    manifest: PackManifest,
    skills_dir: Path,
    output_dir: Path,
    *,
    pack_ids: Sequence[str] | None = None,
    selection_only: bool = False,
) -> tuple[PackReport, ...]:
    """Build selected packs and return their validated reports."""

    manifest = _canonical_manifest(manifest)
    reports = validate_manifest(manifest, skills_dir)
    selected = set(pack_ids or ())
    available = {report.pack.identifier for report in reports}
    unknown = sorted(selected - available)
    if unknown:
        raise PackBuildError("unknown pack selection")

    skills = discover_skills(skills_dir)
    resolved_skills_dir = _resolved_path(skills_dir, "skills directory")
    resolved_output_dir = _resolved_path(output_dir, "pack output")
    if (
        resolved_output_dir == resolved_skills_dir
        or resolved_skills_dir in resolved_output_dir.parents
    ):
        raise PackBuildError("pack output must be outside the skills directory")

    _ensure_directory(output_dir, "pack output")
    selected_reports = tuple(
        report
        for report in reports
        if not selected or report.pack.identifier in selected
    )
    for report in selected_reports:
        _preflight_pack_directory(
            output_dir / report.pack.identifier,
            report,
            skills,
            selection_only=selection_only,
        )

    built: list[PackReport] = []
    for report in selected_reports:
        pack_id = report.pack.identifier
        pack_dir = output_dir / pack_id
        _ensure_directory(pack_dir, f"pack '{pack_id}'")
        _write_pack_metadata(output_dir, manifest, report)

        if not selection_only:
            links_dir = pack_dir / "skills"
            _ensure_directory(links_dir, f"pack '{pack_id}' skills")
            for skill_id in report.pack.skills:
                _link_skill(skills[skill_id], links_dir / skill_id, pack_id)
        built.append(report)
    return tuple(built)


def build_from_files(
    manifest_path: Path = DEFAULT_MANIFEST,
    skills_dir: Path = DEFAULT_SKILLS_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    pack_ids: Sequence[str] | None = None,
    selection_only: bool = False,
) -> tuple[PackReport, ...]:
    """Load, validate, and build packs from repository paths."""

    manifest = load_manifest(manifest_path)
    return build_packs(
        manifest,
        skills_dir,
        output_dir,
        pack_ids=pack_ids,
        selection_only=selection_only,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="versioned pack manifest (default: skills/packs/manifest.json)",
    )
    parser.add_argument(
        "--skills-dir",
        type=Path,
        default=DEFAULT_SKILLS_DIR,
        help="source skill directory (default: skills)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="generated pack directory (default: build/skill-packs)",
    )
    parser.add_argument(
        "--pack",
        dest="pack_ids",
        action="append",
        metavar="ID",
        help="build only this pack; repeat for multiple packs",
    )
    parser.add_argument(
        "--selection-only",
        action="store_true",
        help="write pack metadata without creating skill-directory symlinks",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate the manifest and budgets without writing output",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the pack validator or builder CLI."""

    args = _parser().parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        if args.check:
            reports = validate_manifest(manifest, args.skills_dir)
            print(f"Validated {len(reports)} topical packs.")
            return 0
        built = build_packs(
            manifest,
            args.skills_dir,
            args.output,
            pack_ids=args.pack_ids,
            selection_only=args.selection_only,
        )
    except PackValidationError as exc:
        print("Pack validation failed:", file=sys.stderr)
        for error in exc.errors:
            print(f"- {error}", file=sys.stderr)
        return 2
    except PackBuildError as exc:
        print(f"Pack build failed: {exc}", file=sys.stderr)
        return 2

    for report in built:
        print(
            f"Built {report.pack.identifier}: {report.skill_count} skills, "
            f"{report.size_bytes} bytes."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
