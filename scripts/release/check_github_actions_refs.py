#!/usr/bin/env python3
"""Validate that GitHub Actions workflow references resolve."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import yaml
from yaml.nodes import MappingNode, Node, ScalarNode, SequenceNode

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORKFLOWS_DIR = ROOT / ".github" / "workflows"

FULL_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")


@dataclass(frozen=True)
class ActionRef:
    """A remote action reference found in a workflow file."""

    path: Path
    line_number: int
    spec: str
    repository: str
    ref: str


@dataclass(frozen=True)
class RefAuditResult:
    """Validation result for one action reference."""

    action_ref: ActionRef
    ok: bool
    reason: str


Resolver = Callable[[str, str], tuple[bool, str]]


def workflow_files(workflows_dir: Path = DEFAULT_WORKFLOWS_DIR) -> list[Path]:
    """Return workflow files in deterministic order."""

    return sorted(
        path
        for pattern in ("*.yml", "*.yaml")
        for path in workflows_dir.rglob(pattern)
        if path.is_file()
    )


def parse_action_spec(path: Path, line_number: int, spec: str) -> ActionRef | None:
    """Parse a `uses:` value and return remote GitHub action refs only."""

    if spec.startswith(("./", "../", "docker://")):
        return None
    if "${{" in spec:
        return ActionRef(path, line_number, spec, "", "")
    if "@" not in spec:
        return ActionRef(path, line_number, spec, "", "")

    target, ref = spec.rsplit("@", 1)
    parts = target.split("/")
    if len(parts) < 2 or not parts[0] or not parts[1] or not ref:
        return ActionRef(path, line_number, spec, "", ref)

    repository = "/".join(parts[:2])
    return ActionRef(path, line_number, spec, repository, ref)


def iter_action_refs(
    workflows_dir: Path = DEFAULT_WORKFLOWS_DIR,
) -> Iterable[ActionRef]:
    """Yield remote action references from workflow files."""

    for path in workflow_files(workflows_dir):
        root = yaml.compose(path.read_text(encoding="utf-8"), Loader=yaml.SafeLoader)
        visited: set[int] = set()

        def walk(node: Node | None) -> Iterable[ActionRef]:
            if node is None or id(node) in visited:
                return
            visited.add(id(node))
            if isinstance(node, MappingNode):
                for key, value in node.value:
                    if isinstance(key, ScalarNode) and key.value == "uses":
                        line_number = key.start_mark.line + 1
                        if (
                            not isinstance(value, ScalarNode)
                            or value.tag != "tag:yaml.org,2002:str"
                        ):
                            raise ValueError(
                                f"{path}:{line_number}: uses must be a string"
                            )
                        action_ref = parse_action_spec(
                            path, line_number, value.value.strip()
                        )
                        if action_ref is not None:
                            yield action_ref
                    yield from walk(value)
            elif isinstance(node, SequenceNode):
                for child in node.value:
                    yield from walk(child)

        yield from walk(root)


def resolve_ref(repository: str, ref: str) -> tuple[bool, str]:
    """Return whether a GitHub action tag or branch exists."""

    remote = f"https://github.com/{repository}.git"
    try:
        result = subprocess.run(
            [
                "git",
                "ls-remote",
                "--exit-code",
                "--refs",
                remote,
                f"refs/tags/{ref}",
                f"refs/heads/{ref}",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"could not query {remote}: {exc}"
    if result.returncode == 0:
        return True, "resolved"
    detail = result.stderr.strip() or result.stdout.strip()
    return False, detail or f"{repository}@{ref} did not resolve to a tag or branch"


def audit_action_refs(
    action_refs: Iterable[ActionRef],
    resolver: Resolver = resolve_ref,
    *,
    require_sha: bool = False,
) -> list[RefAuditResult]:
    """Validate action refs with de-duplicated remote lookups."""

    cache: dict[tuple[str, str], tuple[bool, str]] = {}
    results: list[RefAuditResult] = []

    for action_ref in action_refs:
        if not action_ref.repository or not action_ref.ref:
            results.append(
                RefAuditResult(
                    action_ref,
                    False,
                    "remote action refs must be static owner/repo[/path]@ref values",
                )
            )
            continue
        if FULL_SHA_RE.fullmatch(action_ref.ref):
            results.append(RefAuditResult(action_ref, True, "pinned commit SHA"))
            continue
        if require_sha:
            results.append(
                RefAuditResult(
                    action_ref, False, "remote actions must use a full commit SHA"
                )
            )
            continue

        key = (action_ref.repository, action_ref.ref)
        if key not in cache:
            cache[key] = resolver(*key)
        ok, reason = cache[key]
        results.append(RefAuditResult(action_ref, ok, reason))

    return results


def format_result(result: RefAuditResult) -> str:
    """Return a readable single-line audit result."""

    action_ref = result.action_ref
    try:
        relative = action_ref.path.relative_to(ROOT)
    except ValueError:
        relative = action_ref.path
    status = "ok" if result.ok else "fail"
    return (
        f"{status}: {relative}:{action_ref.line_number}: "
        f"{action_ref.spec} ({result.reason})"
    )


def main(argv: list[str] | None = None) -> int:
    """Run the GitHub Actions reference audit."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workflows-dir",
        type=Path,
        default=DEFAULT_WORKFLOWS_DIR,
        help="Directory containing GitHub Actions workflow YAML files.",
    )
    parser.add_argument(
        "--require-sha",
        action="store_true",
        help="Reject mutable tags and branches without performing network lookups.",
    )
    args = parser.parse_args(argv)

    try:
        results = audit_action_refs(
            iter_action_refs(args.workflows_dir), require_sha=args.require_sha
        )
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"GitHub Actions reference policy failed: {exc}", file=sys.stderr)
        return 1
    failures = [result for result in results if not result.ok]

    if failures:
        print("GitHub Actions reference policy failed:", file=sys.stderr)
        for result in failures:
            print(f"- {format_result(result)}", file=sys.stderr)
        return 1

    print("GitHub Actions reference policy passed")
    print(f"- {len(results)} remote action refs resolved or pinned")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
