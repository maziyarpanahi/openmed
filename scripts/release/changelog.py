#!/usr/bin/env python3
"""Render release changelog notes and compute the next SemVer version."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ABOUT_FILE = ROOT / "openmed" / "__about__.py"

VERSION_PATTERN = re.compile(r'__version__\s*=\s*"([^"]+)"')
CONVENTIONAL_PATTERN = re.compile(
    r"^(?P<type>[A-Za-z][A-Za-z0-9-]*)"
    r"(?:\((?P<scope>[^)]+)\))?"
    r"(?P<breaking>!)?: (?P<description>.+)$"
)
BREAKING_PATTERN = re.compile(r"^BREAKING(?: CHANGE|-CHANGE):", re.MULTILINE)

CHANGELOG_SECTIONS = ("Added", "Changed", "Deprecated", "Removed", "Fixed", "Security")
SECTION_BY_TYPE = {
    "feat": "Added",
    "change": "Changed",
    "changed": "Changed",
    "perf": "Changed",
    "refactor": "Changed",
    "deprecate": "Deprecated",
    "deprecated": "Deprecated",
    "remove": "Removed",
    "removed": "Removed",
    "fix": "Fixed",
    "security": "Security",
    "sec": "Security",
}
BUMP_BY_TYPE = {
    "feat": "minor",
    "deprecate": "minor",
    "deprecated": "minor",
    "fix": "patch",
    "perf": "patch",
    "security": "patch",
    "sec": "patch",
}
BUMP_RANK = {"none": 0, "patch": 1, "minor": 2, "major": 3}


@dataclass(frozen=True)
class ConventionalCommit:
    """A parsed Conventional Commit message."""

    subject: str
    body: str = ""
    sha: str = ""
    type: str = ""
    scope: str | None = None
    description: str = ""
    breaking: bool = False


@dataclass(frozen=True)
class ReleaseNotes:
    """Computed release metadata and rendered changelog content."""

    current_version: str
    next_version: str
    bump: str
    markdown: str


def read_package_version(path: Path = ABOUT_FILE) -> str:
    """Read the package version without importing the package."""
    content = path.read_text(encoding="utf-8")
    match = VERSION_PATTERN.search(content)
    if not match:
        raise ValueError(f"Could not find __version__ in {path}")
    return match.group(1)


def parse_conventional_commit(
    subject: str,
    body: str = "",
    sha: str = "",
) -> ConventionalCommit | None:
    """Parse one Conventional Commit subject/body pair."""
    match = CONVENTIONAL_PATTERN.match(subject.strip())
    if not match:
        return None

    commit_type = match.group("type").lower()
    breaking = bool(match.group("breaking")) or bool(BREAKING_PATTERN.search(body))
    return ConventionalCommit(
        subject=subject,
        body=body,
        sha=sha,
        type=commit_type,
        scope=match.group("scope"),
        description=match.group("description").strip(),
        breaking=breaking,
    )


def commit_bump(commit: ConventionalCommit) -> str:
    """Return the SemVer bump implied by one parsed commit."""
    if commit.breaking:
        return "major"
    return BUMP_BY_TYPE.get(commit.type, "none")


def strongest_bump(commits: list[ConventionalCommit]) -> str:
    """Return the strongest SemVer bump implied by parsed commits."""
    bump = "none"
    for commit in commits:
        candidate = commit_bump(commit)
        if BUMP_RANK[candidate] > BUMP_RANK[bump]:
            bump = candidate
    return bump


def bump_version(current_version: str, bump: str) -> str:
    """Apply a SemVer bump to a version string."""
    if bump == "none":
        return current_version

    try:
        major, minor, patch = [int(part) for part in current_version.split(".")]
    except ValueError as exc:
        raise ValueError(
            f"Expected SemVer version X.Y.Z, got {current_version!r}"
        ) from exc

    if bump == "major":
        return f"{major + 1}.0.0"
    if bump == "minor":
        return f"{major}.{minor + 1}.0"
    if bump == "patch":
        return f"{major}.{minor}.{patch + 1}"
    raise ValueError(f"Unknown SemVer bump {bump!r}")


def changelog_section_for(commit: ConventionalCommit) -> str | None:
    """Return the Keep a Changelog section for a parsed commit."""
    return SECTION_BY_TYPE.get(commit.type)


def render_entry(commit: ConventionalCommit) -> str:
    """Render one changelog bullet."""
    prefix = f"**{commit.scope}:** " if commit.scope else ""
    suffix = " (BREAKING)" if commit.breaking else ""
    return f"- {prefix}{commit.description}{suffix}"


def render_changelog(
    version: str,
    commits: list[ConventionalCommit],
    released_on: str | None = None,
) -> str:
    """Render a Keep a Changelog release section from parsed commits."""
    released_on = released_on or date.today().isoformat()
    grouped: dict[str, list[str]] = {section: [] for section in CHANGELOG_SECTIONS}

    for commit in commits:
        section = changelog_section_for(commit)
        if section:
            grouped[section].append(render_entry(commit))

    lines = [f"## [{version}] - {released_on}"]
    for section in CHANGELOG_SECTIONS:
        entries = grouped[section]
        if not entries:
            continue
        lines.extend(["", f"### {section}", "", *entries])

    if len(lines) == 1:
        lines.extend(
            ["", "### Changed", "", "- No user-facing conventional commits found."]
        )

    return "\n".join(lines).rstrip() + "\n"


def build_release_notes(
    current_version: str,
    commits: list[ConventionalCommit],
    released_on: str | None = None,
) -> ReleaseNotes:
    """Compute release metadata and render the changelog section."""
    bump = strongest_bump(commits)
    next_version = bump_version(current_version, bump)
    return ReleaseNotes(
        current_version=current_version,
        next_version=next_version,
        bump=bump,
        markdown=render_changelog(next_version, commits, released_on),
    )


def run_git(repo: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a git command in the selected repository."""
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=60,
    )


def latest_version_tag(repo: Path) -> str | None:
    """Return the most recent reachable v* tag, if any."""
    result = run_git(repo, ["describe", "--tags", "--abbrev=0", "--match", "v[0-9]*"])
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def read_commits(repo: Path, range_spec: str) -> list[ConventionalCommit]:
    """Read and parse Conventional Commits from git log."""
    result = run_git(repo, ["log", "--format=%x1e%H%x1f%s%x1f%b", range_spec])
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git log failed")

    commits: list[ConventionalCommit] = []
    for record in result.stdout.split("\x1e"):
        record = record.strip("\n")
        if not record:
            continue
        parts = record.split("\x1f", 2)
        if len(parts) != 3:
            continue
        parsed = parse_conventional_commit(
            subject=parts[1],
            body=parts[2].strip(),
            sha=parts[0],
        )
        if parsed:
            commits.append(parsed)
    return commits


def default_range_and_version(repo: Path) -> tuple[str, str]:
    """Infer a commit range and base version from the latest reachable tag."""
    tag = latest_version_tag(repo)
    if tag:
        return f"{tag}..HEAD", tag.removeprefix("v")
    return "HEAD", read_package_version()


def write_github_output(outputs: dict[str, str], output_file: Path) -> None:
    """Append outputs using GitHub Actions' multiline output syntax."""
    with output_file.open("a", encoding="utf-8") as handle:
        for key, value in outputs.items():
            if "\n" not in value:
                handle.write(f"{key}={value}\n")
                continue

            delimiter = f"__OPENMED_{key.upper()}__"
            while delimiter in value:
                delimiter += "_"
            value = value if value.endswith("\n") else f"{value}\n"
            handle.write(f"{key}<<{delimiter}\n{value}{delimiter}\n")


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--range",
        dest="range_spec",
        help="Git revision range to scan. Defaults to commits since the latest v* tag.",
    )
    parser.add_argument(
        "--current-version",
        help="Base SemVer version. Defaults to the latest v* tag, then openmed/__about__.py.",
    )
    parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        help="Release date for the rendered changelog section.",
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=ROOT,
        help="Repository path to inspect.",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format for stdout.",
    )
    parser.add_argument(
        "--github-output",
        action="store_true",
        help="Write bump, current_version, next_version, and changelog to GITHUB_OUTPUT.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = parse_args(argv or sys.argv[1:])
    repo = args.repo.resolve()
    range_spec, inferred_version = default_range_and_version(repo)
    range_spec = args.range_spec or range_spec
    current_version = args.current_version or inferred_version

    commits = read_commits(repo, range_spec)
    notes = build_release_notes(current_version, commits, args.date)

    payload = {
        "bump": notes.bump,
        "current_version": notes.current_version,
        "next_version": notes.next_version,
        "changelog": notes.markdown,
    }

    if args.github_output:
        output_path = os.environ.get("GITHUB_OUTPUT")
        if not output_path:
            print("GITHUB_OUTPUT is not set", file=sys.stderr)
            return 2
        write_github_output(payload, Path(output_path))

    if args.format == "json":
        print(json.dumps(payload, indent=2))
    else:
        print(notes.markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
