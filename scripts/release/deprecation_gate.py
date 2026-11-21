"""Require a released deprecation before a breaking major SDK change."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

if __package__:
    from . import api_surface_diff as api
else:
    import api_surface_diff as api


def _version(value):
    if not isinstance(value, str) or not re.fullmatch(
        r"(?:0|[1-9][0-9]*)(?:\.(?:0|[1-9][0-9]*)){1,2}", value
    ):
        raise ValueError("invalid release version")
    numbers = [int(part) for part in value.split(".")]
    return (*numbers, *([0] * (3 - len(numbers))))


def previous_tags(repo: Path, major: int) -> list[str]:
    """Find stable SDK releases of the previous major, newest first."""
    tags = subprocess.check_output(
        ["git", "-C", str(repo), "tag", "--list", f"v{major - 1}.*"], text=True
    )
    stable = [
        tag
        for tag in tags.splitlines()
        if re.fullmatch(r"v(?:0|[1-9][0-9]*)(?:\.(?:0|[1-9][0-9]*)){2}", tag)
    ]
    if not stable:
        raise ValueError("previous_major_tags_missing")
    return sorted(stable, key=lambda tag: _version(tag[1:]), reverse=True)


def python_violations(
    repo: Path, candidate: str, target: tuple, tags: list[str]
) -> list[str]:
    """Compare Python symbols with the latest tag and search released warnings."""
    baseline = api.extract_surface(repo, tags[0], "openmed")
    after = api.extract_surface(repo, candidate, "openmed")
    pending = {change.symbol for change in api.diff_surfaces(baseline, after).breaking}
    for tag in tags:
        if not pending:
            break
        surface = (
            baseline if tag == tags[0] else api.extract_surface(repo, tag, "openmed")
        )
        for name in tuple(pending):
            symbol = surface.get(name)
            if symbol and symbol.deprecated and symbol.remove_in:
                try:
                    due = _version(symbol.remove_in) <= target
                    announced = not symbol.since or _version(symbol.since) <= _version(
                        tag[1:]
                    )
                except ValueError:
                    continue
                if due and announced:
                    pending.remove(name)
    return sorted(pending)


def _openapi_surface(repo: Path, ref: str) -> dict[tuple[str, ...], bool]:
    raw = subprocess.check_output(
        ["git", "-C", str(repo), "show", f"{ref}:docs/api/openapi.json"]
    )
    document = json.loads(raw)
    if not isinstance(document.get("paths"), dict):
        raise ValueError("invalid OpenAPI paths")
    surface = {}

    def resolve(node, seen):
        if not isinstance(node, dict):
            raise ValueError("invalid OpenAPI object")
        while "$ref" in node:
            pointer = node["$ref"]
            if not isinstance(pointer, str) or not pointer.startswith("#/"):
                raise ValueError("local OpenAPI references required")
            if pointer in seen:
                return {}, seen
            seen = seen | {pointer}
            obj = document
            for segment in pointer[2:].split("/"):
                obj = obj[segment.replace("~1", "/").replace("~0", "~")]
            node = {**obj, **{k: v for k, v in node.items() if k != "$ref"}}
        return node, seen

    def properties(schema, prefix, seen=frozenset()):
        if isinstance(schema, bool):
            return
        schema, seen = resolve(schema, seen)
        for name, field in schema.get("properties", {}).items():
            key = (*prefix, name)
            obj, _ = resolve(field, seen) if not isinstance(field, bool) else ({}, seen)
            surface[key] = surface.get(key, True) and obj.get("deprecated") is True
            properties(field, key, seen)
        if "items" in schema:
            properties(schema["items"], (*prefix, "[]"), seen)
        for kind in ("allOf", "anyOf", "oneOf"):
            for part in schema.get(kind, []):
                properties(part, prefix, seen)

    for name, schema in document.get("components", {}).get("schemas", {}).items():
        properties(schema, ("schema", name))
    for path, raw_item in document["paths"].items():
        item, _ = resolve(raw_item, frozenset())
        for method in (
            "get",
            "put",
            "post",
            "delete",
            "patch",
            "options",
            "head",
            "trace",
        ):
            if method not in item:
                continue
            operation = item[method]
            key = ("operation", method.upper(), path)
            surface[key] = operation.get("deprecated") is True
            bodies = [
                ("response " + code, response)
                for code, response in operation.get("responses", {}).items()
            ]
            if "requestBody" in operation:
                bodies.append(("request", operation["requestBody"]))
            for side, body in bodies:
                obj, _ = resolve(body, frozenset())
                for media, content in obj.get("content", {}).items():
                    properties(content.get("schema", {}), (*key, side, media))
    return surface


def openapi_violations(repo: Path, candidate: str, tags: list[str]) -> list[str]:
    """Find removed operations and properties lacking a released warning."""
    baseline = _openapi_surface(repo, tags[0])
    current = _openapi_surface(repo, candidate)
    missing = baseline.keys() - current.keys()
    # Removing an operation or parent property removes its subtree. Require
    # the notice on the outermost removed contract.
    pending = {
        key
        for key in missing
        if not any(key[:i] in missing for i in range(1, len(key)))
    }
    for tag in tags:
        if not pending:
            break
        if tag == tags[0]:
            old = baseline
        else:
            files = subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(repo),
                    "ls-tree",
                    "--name-only",
                    tag,
                    "--",
                    "docs/api/openapi.json",
                ]
            )
            if not files:
                continue
            old = _openapi_surface(repo, tag)
        pending.difference_update(key for key in tuple(pending) if old.get(key) is True)
    return ["REST " + " ".join(key) for key in sorted(pending)]


def run_gate(repo: Path, version: str, candidate: str) -> list[str]:
    """Return breaking symbols lacking a timely deprecation in a released tag."""
    target = _version(version)
    if target[1:] != (0, 0) or target[0] == 0:
        return []
    tags = previous_tags(repo, target[0])
    sha = subprocess.check_output(
        [
            "git",
            "-C",
            str(repo),
            "rev-parse",
            "--verify",
            "--end-of-options",
            f"{candidate}^{{commit}}",
        ],
        text=True,
    ).strip()
    return python_violations(repo, sha, target, tags) + openapi_violations(
        repo, sha, tags
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--candidate", default="HEAD")
    args = parser.parse_args(argv)
    try:
        violations = run_gate(Path.cwd(), args.version, args.candidate)
    except (ValueError, subprocess.CalledProcessError, api.ApiSurfaceError) as exc:
        print(f"Cannot check released deprecations: {exc}", file=sys.stderr)
        return 2
    for name in violations:
        print(f"Missing released deprecation: {name}")
    return int(bool(violations))


if __name__ == "__main__":
    raise SystemExit(main())
