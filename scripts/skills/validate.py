#!/usr/bin/env python3
"""Validate the repository Agent Skills catalog without network access.

The gate checks the local skill contract, markdown references, the committed
skill pack, and executable skill helpers.  It deliberately reports only
repository paths and fixed messages; skill content and subprocess output are
never copied into validation logs.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import unquote, urlsplit

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILLS_ROOT_NAME = "skills"
MARKETPLACE_PATH = Path(".claude-plugin") / "marketplace.json"
HELPER_ROOTS = (Path("skills"), Path("scripts") / "skills")
FOCUSED_TEST_PATH = Path("tests/unit/skills/test_validation.py")
HELPER_TEST_OVERRIDES = {
    Path("skills") / "build_catalog.py": Path("tests/unit/test_skills_catalog.py"),
    Path("scripts") / "skills" / "validate.py": FOCUSED_TEST_PATH,
}
INTERPRETER_HELPER_SUFFIXES = frozenset({".py", ".sh"})

NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
INLINE_LINK_RE = re.compile(
    r"(?<!!)\[[^\]\n]+\]\(\s*(?P<target><[^>\n]*>|[^)\n]+?)\s*\)"
)
REFERENCE_LINK_RE = re.compile(
    r"(?m)^\s{0,3}\[[^\]\n]+\]:\s*(?P<target><[^>\n]*>|[^ \t\n]+)"
)
ALLOWED_PAIRS = {"before", "after", "adjacent"}
MAX_DESCRIPTION_LENGTH = 1024
MAX_BODY_LINES = 500
HELP_TIMEOUT_SECONDS = 10
_YAML_UNAVAILABLE = object()


class ValidationReport:
    """Deterministic validation results and aggregate counts."""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.skill_count = 0
        self.link_count = 0
        self.helper_count = 0

    @property
    def ok(self) -> bool:
        """Return whether the validation completed without errors."""

        return not self.errors


def _display_path(repo_root: Path, path: Path) -> str:
    """Return a stable repository-relative path without exposing file text."""

    root = repo_root.resolve()
    candidate = path.resolve(strict=False)
    try:
        return candidate.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _add_error(
    report: ValidationReport,
    repo_root: Path,
    path: Path,
    message: str,
    *,
    line: int | None = None,
) -> None:
    """Add a path-only error with an optional line number."""

    location = _display_path(repo_root, path)
    if line is not None:
        location = f"{location}:{line}"
    report.errors.append(f"{location}: {message}")


def _read_text(path: Path, report: ValidationReport, repo_root: Path) -> str | None:
    """Read a UTF-8 file, converting all read failures to safe diagnostics."""

    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        _add_error(report, repo_root, path, "file cannot be read as UTF-8")
        return None


def _frontmatter_parts(text: str) -> tuple[str, str] | None:
    """Return raw frontmatter and body when both delimiters are present."""

    lines = text.splitlines(keepends=True)
    if not lines or lines[0].rstrip("\r\n") != "---":
        return None

    for index, line in enumerate(lines[1:], start=1):
        if line.rstrip("\r\n") == "---":
            return "".join(lines[1:index]), "".join(lines[index + 1 :])
    return None


def _parse_yaml_mapping(raw: str) -> dict[str, Any] | object | None:
    """Parse strict YAML while keeping parser details out of diagnostics."""

    try:
        import yaml
    except ModuleNotFoundError:
        return _YAML_UNAVAILABLE

    class UniqueKeyLoader(yaml.SafeLoader):
        """Reject duplicate mapping keys instead of silently overwriting."""

    def construct_mapping(loader: Any, node: Any, deep: bool = False) -> dict[Any, Any]:
        mapping: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in mapping:
                raise yaml.constructor.ConstructorError(
                    None, None, "duplicate key", key_node.start_mark
                )
            mapping[key] = loader.construct_object(value_node, deep=deep)
        return mapping

    UniqueKeyLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping
    )
    try:
        value = yaml.load(raw, Loader=UniqueKeyLoader)
    except (RecursionError, TypeError, ValueError, yaml.YAMLError):
        return None
    return value if isinstance(value, dict) else None


def _validate_frontmatter(
    path: Path,
    text: str,
    report: ValidationReport,
    repo_root: Path,
) -> tuple[dict[str, Any], str] | None:
    """Validate frontmatter shape and return its mapping plus body."""

    if not text.startswith("---"):
        _add_error(report, repo_root, path, "frontmatter is missing")
        return None

    parts = _frontmatter_parts(text)
    if parts is None:
        _add_error(report, repo_root, path, "frontmatter delimiters are invalid")
        return None

    raw, body = parts
    metadata = _parse_yaml_mapping(raw)
    if metadata is _YAML_UNAVAILABLE:
        _add_error(report, repo_root, path, "YAML parser is unavailable")
        return None
    if metadata is None:
        _add_error(report, repo_root, path, "frontmatter is not valid YAML")
        return None

    return metadata, body


def _validate_metadata(
    path: Path,
    metadata: dict[str, Any],
    report: ValidationReport,
    repo_root: Path,
) -> None:
    """Validate the small set of catalog metadata identifiers used in skills."""

    nested = metadata.get("metadata")
    if nested is None:
        return
    if not isinstance(nested, dict):
        _add_error(report, repo_root, path, "metadata must be a mapping")
        return

    for key in ("project", "category", "pairs", "version"):
        if key in nested and not isinstance(nested[key], str):
            _add_error(report, repo_root, path, "metadata value has an invalid type")

    pairs = nested.get("pairs")
    if isinstance(pairs, str) and pairs and pairs not in ALLOWED_PAIRS:
        _add_error(report, repo_root, path, "metadata pairs identifier is invalid")

    category = nested.get("category")
    if isinstance(category, str) and category and not NAME_RE.fullmatch(category):
        _add_error(report, repo_root, path, "metadata category identifier is invalid")


def _validate_identifier(
    path: Path,
    metadata: dict[str, Any],
    report: ValidationReport,
    repo_root: Path,
    seen: set[str],
) -> str:
    """Validate the skill name and return a safe folder-derived fallback."""

    folder_name = path.parent.name
    name = metadata.get("name")
    if not isinstance(name, str) or not name.strip():
        _add_error(report, repo_root, path, "skill identifier is missing")
        return folder_name

    if name != folder_name:
        _add_error(report, repo_root, path, "skill identifier does not match folder")
    if NAME_RE.fullmatch(name) is None:
        _add_error(report, repo_root, path, "skill identifier is not kebab-case")
    if name in seen:
        _add_error(report, repo_root, path, "skill identifier is duplicated")
    seen.add(name)
    return name


def _validate_skill_text(
    path: Path,
    text: str,
    report: ValidationReport,
    repo_root: Path,
    seen_names: set[str],
) -> None:
    """Validate one ``SKILL.md`` and its local markdown references."""

    parsed = _validate_frontmatter(path, text, report, repo_root)
    if parsed is None:
        return

    metadata, body = parsed
    _validate_identifier(path, metadata, report, repo_root, seen_names)

    description = metadata.get("description")
    if not isinstance(description, str) or not description.strip():
        _add_error(report, repo_root, path, "skill description is missing")
    elif len(description) > MAX_DESCRIPTION_LENGTH:
        _add_error(report, repo_root, path, "skill description is too long")

    license_name = metadata.get("license")
    if license_name is not None and (
        not isinstance(license_name, str) or not license_name.strip()
    ):
        _add_error(report, repo_root, path, "license must be a non-empty string")

    _validate_metadata(path, metadata, report, repo_root)

    if len(body.splitlines()) > MAX_BODY_LINES:
        _add_error(report, repo_root, path, "skill body exceeds 500 lines")
    if not body.strip():
        _add_error(report, repo_root, path, "skill body is empty")

    _validate_internal_links(path, body, report, repo_root)


def _link_target(raw_target: str) -> str | None:
    """Extract a markdown destination without interpreting its label text."""

    target = raw_target.strip()
    if target.startswith("<"):
        closing = target.find(">")
        if closing < 0:
            return None
        target = target[1:closing]
    else:
        target = target.split(maxsplit=1)[0] if target else ""
    return unquote(target)


def _local_target(raw_target: str) -> str | None:
    """Return a relative local path, or ``None`` for external/anchor links."""

    target = _link_target(raw_target)
    if not target:
        return None
    parsed = urlsplit(target)
    if parsed.scheme or parsed.netloc or target.startswith("//"):
        return None
    if not parsed.path:
        return None
    return parsed.path


def _markdown_links(body: str) -> list[tuple[int, str]]:
    """Return local-link candidates as source line and raw destination pairs."""

    matches: list[tuple[int, str]] = []
    for pattern in (INLINE_LINK_RE, REFERENCE_LINK_RE):
        for match in pattern.finditer(body):
            target = match.group("target")
            line = body.count("\n", 0, match.start()) + 1
            matches.append((line, target))
    return sorted(set(matches))


def _validate_internal_links(
    path: Path,
    body: str,
    report: ValidationReport,
    repo_root: Path,
) -> None:
    """Resolve local markdown links without making any network request."""

    for line, raw_target in _markdown_links(body):
        target = _local_target(raw_target)
        if target is None:
            continue
        report.link_count += 1
        candidate = (path.parent / Path(target)).resolve(strict=False)
        if not candidate.is_relative_to(repo_root.resolve()):
            _add_error(
                report,
                repo_root,
                path,
                "internal link escapes the repository",
                line=line,
            )
        elif not candidate.exists():
            _add_error(
                report,
                repo_root,
                path,
                "internal link target is missing",
                line=line,
            )


def _pack_skill_name(entry: object) -> str | None:
    """Parse the canonical ``./skills/<identifier>`` marketplace form."""

    if not isinstance(entry, str) or not entry.startswith("./skills/"):
        return None
    name = entry.removeprefix("./skills/")
    if not name or "/" in name or "\\" in name or NAME_RE.fullmatch(name) is None:
        return None
    return name


def _validate_pack_membership(
    repo_root: Path,
    skill_names: set[str],
    report: ValidationReport,
) -> None:
    """Ensure the marketplace pack contains each catalog skill exactly once."""

    marketplace = repo_root / MARKETPLACE_PATH
    if not marketplace.is_file():
        _add_error(report, repo_root, marketplace, "skill pack manifest is missing")
        return

    try:
        payload = json.loads(marketplace.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        _add_error(
            report, repo_root, marketplace, "skill pack manifest is invalid JSON"
        )
        return

    if not isinstance(payload, dict) or not isinstance(payload.get("plugins"), list):
        _add_error(report, repo_root, marketplace, "skill pack list is missing")
        return

    entries: list[str] = []
    for plugin in payload["plugins"]:
        if not isinstance(plugin, dict) or not isinstance(plugin.get("skills"), list):
            _add_error(report, repo_root, marketplace, "skill pack entry is invalid")
            continue
        for entry in plugin["skills"]:
            name = _pack_skill_name(entry)
            if name is None:
                _add_error(report, repo_root, marketplace, "skill pack path is invalid")
            else:
                entries.append(name)

    counts: dict[str, int] = {}
    for name in entries:
        counts[name] = counts.get(name, 0) + 1

    for name in sorted(counts):
        if counts[name] > 1:
            _add_error(
                report,
                repo_root,
                repo_root / SKILLS_ROOT_NAME / name / "SKILL.md",
                "skill has duplicate pack membership",
            )

    pack_names = set(entries)
    for name in sorted(skill_names - pack_names):
        _add_error(
            report,
            repo_root,
            repo_root / SKILLS_ROOT_NAME / name / "SKILL.md",
            "skill is not present in a pack",
        )
    for name in sorted(pack_names - skill_names):
        _add_error(
            report,
            repo_root,
            marketplace,
            "skill pack contains an unknown skill",
        )


def _helper_command(path: Path) -> list[str]:
    """Build a non-shell help command for a local executable helper."""

    if path.suffix == ".py":
        return [sys.executable, str(path), "--help"]
    if path.suffix == ".sh":
        return ["bash", str(path), "--help"]
    return [str(path), "--help"]


def _executable_helpers(repo_root: Path) -> list[Path]:
    """Find executable helper files in the skill-owned directories."""

    helpers: list[Path] = []
    for relative_root in HELPER_ROOTS:
        root = repo_root / relative_root
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and (
                path.stat().st_mode & 0o111
                or path.suffix.lower() in INTERPRETER_HELPER_SUFFIXES
            ):
                helpers.append(path)
    return helpers


def _validate_helper_help(
    path: Path,
    report: ValidationReport,
    repo_root: Path,
) -> None:
    """Run a helper's help command with local-only environment flags."""

    env = os.environ.copy()
    env.update(
        {
            "OPENMED_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONHASHSEED": "0",
        }
    )
    try:
        result = subprocess.run(
            _helper_command(path),
            cwd=repo_root,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=HELP_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        _add_error(report, repo_root, path, "--help command timed out")
        return
    except OSError:
        _add_error(report, repo_root, path, "--help command could not run")
        return

    if result.returncode != 0:
        _add_error(report, repo_root, path, "--help command failed")


def _focused_test_for_helper(repo_root: Path, helper: Path) -> Path:
    """Return the conventional focused test path for an executable helper."""

    relative = helper.relative_to(repo_root)
    return HELPER_TEST_OVERRIDES.get(
        relative,
        Path("tests/unit/skills") / f"test_{helper.stem}.py",
    )


def _validate_helper_test(
    helper: Path,
    report: ValidationReport,
    repo_root: Path,
) -> None:
    """Require a parseable focused test file for each executable helper."""

    test_path = repo_root / _focused_test_for_helper(repo_root, helper)
    if not test_path.is_file():
        _add_error(report, repo_root, test_path, "focused helper test is missing")
        return

    try:
        tree = ast.parse(test_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, SyntaxError):
        _add_error(report, repo_root, test_path, "focused helper test is invalid")
        return

    has_test = any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
        for node in ast.walk(tree)
    )
    if not has_test:
        _add_error(report, repo_root, test_path, "focused helper test is empty")


def validate_repository(
    repo_root: Path | None = None,
    *,
    run_helper_help: bool = True,
) -> ValidationReport:
    """Validate the local skill catalog and return a deterministic report.

    Args:
        repo_root: Repository root to inspect. Defaults to this checkout.
        run_helper_help: Probe executable helper ``--help`` commands. Tests for
            malformed synthetic repositories can disable subprocess probes.
    """

    root = (repo_root or REPO_ROOT).resolve()
    report = ValidationReport()
    skills_root = root / SKILLS_ROOT_NAME
    skill_names: set[str] = set()
    seen_names: set[str] = set()

    if not skills_root.is_dir():
        _add_error(report, root, skills_root, "skills directory is missing")
    else:
        children = sorted(
            (path for path in skills_root.iterdir() if path.is_dir()),
            key=lambda path: path.name,
        )
        for child in children:
            if child.name.startswith((".", "_")):
                continue
            skill_names.add(child.name)
            skill_path = child / "SKILL.md"
            if not skill_path.is_file():
                _add_error(report, root, skill_path, "SKILL.md is missing")
                continue
            if NAME_RE.fullmatch(child.name) is None:
                _add_error(
                    report, root, skill_path, "skill folder identifier is invalid"
                )
            text = _read_text(skill_path, report, root)
            if text is None:
                continue
            _validate_skill_text(skill_path, text, report, root, seen_names)
            report.skill_count += 1

    _validate_pack_membership(root, skill_names, report)

    helpers = _executable_helpers(root)
    report.helper_count = len(helpers)
    if run_helper_help:
        for helper in helpers:
            _validate_helper_help(helper, report, root)
            _validate_helper_test(helper, report, root)

    report.errors = sorted(set(report.errors))
    return report


def format_report(report: ValidationReport) -> str:
    """Format a report without including skill content or subprocess output."""

    if report.errors:
        lines = ["Agent Skills validation failed:"]
        lines.extend(f"- {error}" for error in report.errors)
        return "\n".join(lines)
    return (
        "Agent Skills validation passed: "
        f"{report.skill_count} skills, {report.link_count} local links, "
        f"{report.helper_count} executable helpers."
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the offline validation command-line parser."""

    parser = argparse.ArgumentParser(
        description="Validate the local Agent Skills catalog without network access."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root to validate (defaults to the current checkout).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate only; retained as an explicit CI-friendly alias.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the validation CLI and return a process exit code."""

    args = build_parser().parse_args(argv)
    report = validate_repository(args.repo_root)
    stream = sys.stderr if report.errors else sys.stdout
    print(format_report(report), file=stream)
    return 1 if report.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
