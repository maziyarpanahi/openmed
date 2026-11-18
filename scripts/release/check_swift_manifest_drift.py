#!/usr/bin/env python3
"""Check that the root and nested OpenMedKit SwiftPM manifests agree.

Consumers that add the repository URL resolve the root ``Package.swift``. The
demos and the iOS, watchOS and visionOS CI lanes build
``swift/OpenMedKit/Package.swift``. Both manifests are hand-written; the root
manifest is the consumer-facing source of truth and the nested manifest mirrors
it. This check compares normalized ``swift package dump-package`` output so the
two may differ only in where target sources live: root target paths carry the
``swift/OpenMedKit/`` prefix, nested paths are relative to the nested package
and may rely on SwiftPM's default ``Sources/<name>`` and ``Tests/<name>``
locations.

It also checks that the committed root ``Package.resolved`` pins a version that
satisfies every dependency requirement both manifests agree on, so a manifest bump
cannot land with a stale lock file.

The report lists only normalized manifest facts (dependency identities and
version requirements, product and target names, conditions and relative source
paths). It never prints absolute paths or ``swift`` diagnostics.

Usage::

    python3 scripts/release/check_swift_manifest_drift.py
    python3 scripts/release/check_swift_manifest_drift.py \\
        --root-dump root.json --nested-dump nested.json

Exit codes: 0 when the manifests agree, 1 on drift or a stale
``Package.resolved``, 2 when an input cannot be dumped or parsed.
"""

from __future__ import annotations

import argparse
import json
import posixpath
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
NESTED_PACKAGE = PurePosixPath("swift/OpenMedKit")
ROOT_SOURCE_PREFIX = NESTED_PACKAGE

SECTIONS = ("package", "dependencies", "products", "targets")
_PACKAGE_SKIP_KEYS = frozenset(
    {"dependencies", "packageKind", "platforms", "products", "targets"}
)
_TARGET_STRUCTURED_KEYS = frozenset(
    {"dependencies", "exclude", "name", "path", "resources", "sources", "type"}
)
_DEFAULT_TARGET_DIRECTORIES = {"test": "Tests", "plugin": "Plugins"}
_SEMVER_RE = re.compile(
    r"^(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z.-]+))?(?:\+[0-9A-Za-z.-]+)?$"
)

Runner = Callable[..., subprocess.CompletedProcess[str]]


class ManifestDumpError(RuntimeError):
    """A manifest could not be dumped or parsed.

    The message is a controlled reason code plus the manifest role, never a
    path or raw ``swift`` diagnostics.
    """


@dataclass(frozen=True)
class SectionDiff:
    """Normalized facts that appear in only one manifest for one section.

    Attributes:
        section: One of ``package``, ``dependencies``, ``products`` or
            ``targets``.
        root_only: Lines present only in the root (consumer-facing) manifest.
        nested_only: Lines present only in the nested manifest.
    """

    section: str
    root_only: tuple[str, ...]
    nested_only: tuple[str, ...]


def _compact(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _first(value: Any) -> Any:
    if isinstance(value, list) and value:
        return value[0]
    return value


def _render_requirement(requirement: Mapping[str, Any] | None) -> str:
    if not requirement:
        return "unversioned"
    kind, value = next(iter(sorted(requirement.items())))
    value = _first(value)
    if kind == "range" and isinstance(value, Mapping):
        return f"range {value.get('lowerBound')}..<{value.get('upperBound')}"
    if isinstance(value, str):
        return f"{kind} {value}"
    return f"{kind} {_compact(value)}"


def _render_location(kind: str, details: Mapping[str, Any]) -> str:
    location = details.get("location")
    if isinstance(location, Mapping):
        remote = location.get("remote")
        if remote:
            entry = _first(remote)
            if isinstance(entry, Mapping):
                return f"{kind} {entry.get('urlString')}"
            return f"{kind} {entry}"
        if location.get("local"):
            return f"{kind} local"
    if kind == "fileSystem":
        return "fileSystem local"
    return kind


def _dependency_details(dump: Mapping[str, Any]) -> dict[str, tuple[str, Any]]:
    """Return ``identity -> (kind, details)`` for package dependencies."""

    entries: dict[str, tuple[str, Any]] = {}
    for dependency in dump.get("dependencies") or []:
        kind, raw = next(iter(dependency.items()))
        details = _first(raw)
        if not isinstance(details, Mapping):
            raise ManifestDumpError("unrecognized_dependency_entry")
        identity = str(details.get("identity") or "").lower()
        if not identity:
            raise ManifestDumpError("dependency_without_identity")
        entries[identity] = (kind, details)
    return entries


def _dependency_facts(dump: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    """Return ``identity -> {aspect: rendered value}`` for package dependencies."""

    facts: dict[str, dict[str, str]] = {}
    for identity, (kind, details) in _dependency_details(dump).items():
        aspects = {
            "requirement": _render_requirement(details.get("requirement")),
            "location": _render_location(kind, details),
        }
        traits = details.get("traits")
        if traits not in (None, [], [{"name": "default"}]):
            aspects["traits"] = _compact(traits)
        if details.get("productFilter") is not None:
            aspects["productFilter"] = _compact(details["productFilter"])
        facts[identity] = aspects
    return facts


def _render_product_type(product_type: Any) -> str:
    if isinstance(product_type, Mapping) and product_type:
        kind, value = next(iter(sorted(product_type.items())))
        value = _first(value)
        if value in (None, {}, []):
            return str(kind)
        if isinstance(value, str):
            return f"{kind}({value})"
        return f"{kind}({_compact(value)})"
    return str(product_type)


def _render_condition(condition: Any) -> str:
    if not condition:
        return ""
    if not isinstance(condition, Mapping):
        return f" when {_compact(condition)}"
    parts = []
    for key, value in sorted(condition.items()):
        if value in (None, []):
            continue
        label = "platforms" if key == "platformNames" else key
        if isinstance(value, list):
            parts.append(f"{label}={','.join(sorted(str(item) for item in value))}")
        else:
            parts.append(f"{label}={_compact(value)}")
    return f" when {' '.join(parts)}" if parts else ""


def _render_target_dependency(dependency: Mapping[str, Any]) -> str:
    kind, raw = next(iter(dependency.items()))
    values = list(raw) if isinstance(raw, list) else [raw]
    if kind == "product":
        name = values[0]
        package = values[1] if len(values) > 1 else None
        aliases = values[2] if len(values) > 2 else None
        condition = values[3] if len(values) > 3 else None
        rendered = f"product:{name}@{package}"
        if aliases:
            rendered += f" aliases={_compact(aliases)}"
        return rendered + _render_condition(condition)
    name = values[0]
    condition = values[1] if len(values) > 1 else None
    return f"{kind}:{name}{_render_condition(condition)}"


def _normalize_path(
    path: str | None, *, name: str, target_type: str, source_prefix: PurePosixPath
) -> str:
    if path is None:
        directory = _DEFAULT_TARGET_DIRECTORIES.get(target_type, "Sources")
        path = f"{directory}/{name}"
    normalized = posixpath.normpath(path)
    if source_prefix.parts:
        # Express root paths relative to the nested package, so a root target
        # outside swift/OpenMedKit renders as ``../../...`` and never matches.
        normalized = posixpath.relpath(normalized, source_prefix.as_posix())
    return normalized


def _render_resource(resource: Mapping[str, Any]) -> str:
    rule = resource.get("rule")
    rule_text = _render_product_type(rule) if rule else "unknown"
    return f"resource {rule_text} {resource.get('path')}"


def _target_lines(
    target: Mapping[str, Any], *, source_prefix: PurePosixPath
) -> list[str]:
    name = str(target.get("name"))
    target_type = str(target.get("type"))
    lines = [
        f"{name} type {target_type}",
        "{} path {}".format(
            name,
            _normalize_path(
                target.get("path"),
                name=name,
                target_type=target_type,
                source_prefix=source_prefix,
            ),
        ),
    ]
    for dependency in target.get("dependencies") or []:
        lines.append(f"{name} depends {_render_target_dependency(dependency)}")
    for resource in target.get("resources") or []:
        lines.append(f"{name} {_render_resource(resource)}")
    for excluded in target.get("exclude") or []:
        lines.append(f"{name} exclude {excluded}")
    sources = target.get("sources")
    if sources is not None:
        lines.append(f"{name} sources {_compact(sorted(sources))}")
    for key, value in sorted(target.items()):
        if key in _TARGET_STRUCTURED_KEYS or value in (None, [], {}):
            continue
        lines.append(f"{name} {key} {_compact(value)}")
    return lines


def normalize_manifest(
    dump: Mapping[str, Any], *, source_prefix: PurePosixPath | str = ""
) -> dict[str, list[str]]:
    """Normalize ``swift package dump-package`` output into comparable lines.

    Args:
        dump: Parsed JSON from ``swift package dump-package``.
        source_prefix: Location of the nested package inside this package.
            Target paths, explicit or SwiftPM defaults, are rendered relative to
            it, so a root target at ``swift/OpenMedKit/Sources/X`` compares
            equal to a nested target at ``Sources/X`` or one using SwiftPM's
            default location, while a root target elsewhere renders as
            ``../../...`` and never matches.

    Returns:
        A mapping from section name to a sorted list of normalized facts. The
        ``dependencies`` section renders each package dependency as
        ``<identity> <requirement>`` plus ``<identity> location <source>``.

    Raises:
        ManifestDumpError: If a dependency entry has no identity or an
            unrecognized shape.
    """

    prefix = PurePosixPath(source_prefix) if source_prefix else PurePosixPath()
    package_lines = []
    for key, value in sorted(dump.items()):
        if key in _PACKAGE_SKIP_KEYS:
            continue
        package_lines.append(f"{key} {_compact(value)}")
    for platform in dump.get("platforms") or []:
        line = f"platform {platform.get('platformName')} {platform.get('version')}"
        if platform.get("options"):
            line += f" options={_compact(platform['options'])}"
        package_lines.append(line)

    dependency_lines = []
    for identity, aspects in _dependency_facts(dump).items():
        dependency_lines.append(f"{identity} {aspects['requirement']}")
        for aspect, value in sorted(aspects.items()):
            if aspect != "requirement":
                dependency_lines.append(f"{identity} {aspect} {value}")

    product_lines = []
    for product in dump.get("products") or []:
        targets = ",".join(sorted(str(item) for item in product.get("targets") or []))
        line = (
            f"{product.get('name')} {_render_product_type(product.get('type'))} "
            f"targets={targets}"
        )
        if product.get("settings"):
            line += f" settings={_compact(product['settings'])}"
        product_lines.append(line)

    target_lines = []
    for target in dump.get("targets") or []:
        target_lines.extend(_target_lines(target, source_prefix=prefix))

    return {
        "package": sorted(package_lines),
        "dependencies": sorted(dependency_lines),
        "products": sorted(product_lines),
        "targets": sorted(target_lines),
    }


def _dependency_section_diff(
    root: Mapping[str, Any], nested: Mapping[str, Any]
) -> SectionDiff:
    root_facts = _dependency_facts(root)
    nested_facts = _dependency_facts(nested)
    root_only: list[str] = []
    nested_only: list[str] = []
    for identity in sorted(set(root_facts) | set(nested_facts)):
        left = root_facts.get(identity)
        right = nested_facts.get(identity)
        if left is None or right is None:
            present, bucket = (
                (left, root_only) if right is None else (right, nested_only)
            )
            assert present is not None
            bucket.append(f"{identity} {present['requirement']}")
            continue
        for aspect in sorted(set(left) | set(right)):
            if left.get(aspect) == right.get(aspect):
                continue
            label = "" if aspect == "requirement" else f"{aspect} "
            if aspect in left:
                root_only.append(f"{identity} {label}{left[aspect]}")
            if aspect in right:
                nested_only.append(f"{identity} {label}{right[aspect]}")
    return SectionDiff("dependencies", tuple(root_only), tuple(nested_only))


def compare_manifests(
    root: Mapping[str, Any],
    nested: Mapping[str, Any],
    *,
    root_source_prefix: PurePosixPath | str = ROOT_SOURCE_PREFIX,
) -> list[SectionDiff]:
    """Compare a root and a nested manifest dump.

    Args:
        root: ``dump-package`` JSON for the repository-root manifest.
        nested: ``dump-package`` JSON for ``swift/OpenMedKit/Package.swift``.
        root_source_prefix: Location of the nested package inside the root
            package; root target paths are rendered relative to it.

    Returns:
        One ``SectionDiff`` per section that disagrees, in ``SECTIONS`` order.
        An empty list means the manifests agree. Dependencies are reported by
        identity, so a changed requirement in one manifest yields only
        ``<identity> <requirement>`` lines.
    """

    root_sections = normalize_manifest(root, source_prefix=root_source_prefix)
    nested_sections = normalize_manifest(nested)
    diffs = []
    for section in SECTIONS:
        if section == "dependencies":
            diff = _dependency_section_diff(root, nested)
        else:
            left = set(root_sections[section])
            right = set(nested_sections[section])
            diff = SectionDiff(
                section,
                tuple(sorted(left - right)),
                tuple(sorted(right - left)),
            )
        if diff.root_only or diff.nested_only:
            diffs.append(diff)
    return diffs


def _semver_key(version: str) -> tuple[int, int, int, int, str] | None:
    match = _SEMVER_RE.match(version)
    if match is None:
        return None
    major, minor, patch, prerelease = match.groups()
    return (
        int(major),
        int(minor),
        int(patch),
        0 if prerelease else 1,
        prerelease or "",
    )


def _pin_satisfies(requirement: Mapping[str, Any], state: Mapping[str, Any]) -> bool:
    kind, value = next(iter(sorted(requirement.items())))
    value = _first(value)
    version = state.get("version")
    if kind == "exact":
        return version == value
    if kind == "range" and isinstance(value, Mapping):
        pinned = _semver_key(str(version)) if version else None
        lower = _semver_key(str(value.get("lowerBound")))
        upper = _semver_key(str(value.get("upperBound")))
        if pinned is None or lower is None or upper is None:
            return False
        return lower <= pinned < upper
    if kind == "revision":
        return state.get("revision") == value
    if kind == "branch":
        return state.get("branch") == value
    return True


def check_resolved_pins(
    root: Mapping[str, Any],
    nested: Mapping[str, Any],
    resolved: Mapping[str, Any],
) -> list[str]:
    """Check root ``Package.resolved`` pins against agreed manifest requirements.

    Only dependencies that both manifests declare identically are checked; a
    dependency the manifests disagree on is already reported as drift.

    Args:
        root: ``dump-package`` JSON for the repository-root manifest.
        nested: ``dump-package`` JSON for the nested manifest.
        resolved: Parsed root ``Package.resolved`` (format version 2 or 3).

    Returns:
        One ``<identity> <requirement> pinned <version>`` line per dependency
        whose pin is missing or does not satisfy the requirement.
    """

    root_facts = _dependency_facts(root)
    nested_facts = _dependency_facts(nested)
    root_details = _dependency_details(root)
    pins = {
        str(pin.get("identity") or "").lower(): pin.get("state") or {}
        for pin in resolved.get("pins") or []
        if isinstance(pin, Mapping)
    }
    problems = []
    for identity, aspects in sorted(root_facts.items()):
        requirement = root_details[identity][1].get("requirement")
        if nested_facts.get(identity) != aspects or not requirement:
            continue
        state = pins.get(identity)
        if state is None:
            problems.append(f"{identity} {aspects['requirement']} pinned nothing")
        elif not _pin_satisfies(requirement, state):
            pinned = (
                state.get("version") or state.get("branch") or state.get("revision")
            )
            problems.append(f"{identity} {aspects['requirement']} pinned {pinned}")
    return problems


def format_report(
    diffs: Sequence[SectionDiff], resolved_problems: Sequence[str] = ()
) -> str:
    """Render a drift report.

    Args:
        diffs: Output of ``compare_manifests``.
        resolved_problems: Output of ``check_resolved_pins``.

    Returns:
        A human-readable report. Lines prefixed ``-`` appear only in the root
        manifest, lines prefixed ``+`` appear only in the nested manifest and
        lines prefixed ``!`` are stale ``Package.resolved`` pins.
    """

    if not diffs and not resolved_problems:
        return (
            "Swift manifests agree: Package.swift and "
            f"{NESTED_PACKAGE}/Package.swift differ only in source paths."
        )
    lines: list[str] = []
    if diffs:
        lines.extend(
            [
                f"Swift manifest drift: Package.swift (root, consumer-facing) and "
                f"{NESTED_PACKAGE}/Package.swift (nested) disagree.",
                "'-' lines appear only in the root manifest; '+' lines only in the "
                "nested manifest. Update both manifests together.",
            ]
        )
        for diff in diffs:
            lines.append("")
            lines.append(f"[{diff.section}]")
            lines.extend(f"- {line}" for line in diff.root_only)
            lines.extend(f"+ {line}" for line in diff.nested_only)
    if resolved_problems:
        if lines:
            lines.append("")
        lines.append(
            "Package.resolved is stale: run `swift package resolve` at the "
            "repository root and commit the result."
        )
        lines.append("")
        lines.append("[Package.resolved]")
        lines.extend(f"! {line}" for line in resolved_problems)
    return "\n".join(lines)


def dump_manifest(
    package_dir: Path,
    *,
    role: str,
    swift: str = "swift",
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    """Run ``swift package dump-package`` for one package directory.

    Args:
        package_dir: Directory containing ``Package.swift``.
        role: ``root`` or ``nested``; used in error reason codes.
        swift: The ``swift`` executable.
        runner: Injected ``subprocess.run`` replacement for tests.

    Returns:
        The parsed manifest dump.

    Raises:
        ManifestDumpError: If ``swift`` is missing, exits non-zero or prints
            something other than a JSON object.
    """

    command = [swift, "package", "--package-path", str(package_dir), "dump-package"]
    try:
        completed = runner(command, capture_output=True, text=True, check=False)
    except FileNotFoundError as error:
        raise ManifestDumpError(f"swift_unavailable role={role}") from error
    if completed.returncode != 0:
        raise ManifestDumpError(
            f"dump_package_failed role={role} exit={completed.returncode}"
        )
    return _parse_dump(completed.stdout, role=role)


def _parse_dump(text: str, *, role: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as error:
        raise ManifestDumpError(f"dump_not_json role={role}") from error
    if not isinstance(parsed, dict):
        raise ManifestDumpError(f"dump_not_object role={role}")
    return parsed


def _load_dump(path: Path, *, role: str) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ManifestDumpError(f"dump_unreadable role={role}") from error
    return _parse_dump(text, role=role)


def main(argv: Sequence[str] | None = None, *, runner: Runner = subprocess.run) -> int:
    """Run the manifest drift check.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.
        runner: Injected ``subprocess.run`` replacement for tests.

    Returns:
        ``0`` when the manifests agree, ``1`` on drift or a stale
        ``Package.resolved`` and ``2`` when an input cannot be dumped or parsed.
    """

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--swift", default="swift", help="swift executable")
    parser.add_argument(
        "--root-dump", type=Path, help="pre-captured root dump-package JSON"
    )
    parser.add_argument(
        "--nested-dump", type=Path, help="pre-captured nested dump-package JSON"
    )
    parser.add_argument(
        "--resolved",
        type=Path,
        help=(
            "root Package.resolved to check; defaults to the repository's when "
            "the manifests are dumped live"
        ),
    )
    args = parser.parse_args(argv)
    resolved_path = args.resolved
    if resolved_path is None and args.root_dump is None:
        resolved_path = args.repo_root / "Package.resolved"

    try:
        if args.root_dump is not None:
            root = _load_dump(args.root_dump, role="root")
        else:
            root = dump_manifest(
                args.repo_root, role="root", swift=args.swift, runner=runner
            )
        if args.nested_dump is not None:
            nested = _load_dump(args.nested_dump, role="nested")
        else:
            nested = dump_manifest(
                args.repo_root / NESTED_PACKAGE,
                role="nested",
                swift=args.swift,
                runner=runner,
            )
        diffs = compare_manifests(root, nested)
        resolved_problems: list[str] = []
        if resolved_path is not None:
            resolved = _load_dump(resolved_path, role="resolved")
            resolved_problems = check_resolved_pins(root, nested, resolved)
    except ManifestDumpError as error:
        print(f"Swift manifest drift check could not run: {error}", file=sys.stderr)
        return 2

    report = format_report(diffs, resolved_problems)
    if diffs or resolved_problems:
        print(report, file=sys.stderr)
        return 1
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
