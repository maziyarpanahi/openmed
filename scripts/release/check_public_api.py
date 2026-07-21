#!/usr/bin/env python3
"""Guard the public API surface against unannounced backwards-incompatible changes.

This tool snapshots the public API surface of ``openmed`` -- the names exported
through ``openmed.__all__`` plus the ``__all__`` exports of its public
subpackages, together with the signatures of public callables -- into a
committed baseline JSON file. On every run it recaptures the current surface and
diffs it against the baseline.

Changes are classified into two buckets:

* **Additions** -- a new exported name, or a new module gaining exports. These
  are always backwards compatible and never fail the check.
* **Breaking changes** -- a removed export, a member changing kind (for example a
  function becoming a class), or a callable losing/renaming/reordering
  parameters or making a previously optional parameter required. These fail the
  check *unless* they are recorded as intentional in the deprecation allowlist.

The deprecation policy lives in ``docs/compliance/api-deprecation-policy.md`` and
intentional breaks are recorded in ``scripts/release/public_api_allowlist.json``.
Regenerate the baseline with ``--update`` once a break has been announced and
allowlisted, or when adding new public surface.

The implementation is stdlib-only (``ast``/``inspect``) so it can run in the
``repo-policy`` CI job without importing heavy optional dependencies.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE = ROOT / "scripts" / "release" / "public_api_baseline.json"
DEFAULT_ALLOWLIST = ROOT / "scripts" / "release" / "public_api_allowlist.json"

TOP_LEVEL_PACKAGE = "openmed"

# Public subpackages whose ``__all__`` exports form part of the supported
# surface. Private helpers, backend shims, and dependency-heavy internals are
# intentionally excluded; only stable, documented namespaces belong here.
PUBLIC_SUBPACKAGES: tuple[str, ...] = (
    "openmed.core",
    "openmed.processing",
    "openmed.utils",
    "openmed.clinical",
    "openmed.eval",
    "openmed.risk",
    "openmed.interop",
    "openmed.structured",
    "openmed.mlx",
    "openmed.multimodal",
    "openmed.ner",
    "openmed.zero_shot",
    "openmed.compliance",
)

# Schema version for the baseline document. Bump when the capture format changes
# in a way that older baselines cannot be diffed against.
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class MemberSnapshot:
    """A captured public member: its kind and, for callables, its signature."""

    name: str
    kind: str
    signature: str | None


@dataclass(frozen=True)
class SurfaceSnapshot:
    """The captured public surface for a single module namespace."""

    module: str
    members: tuple[MemberSnapshot, ...]


@dataclass(frozen=True)
class Change:
    """A single classified difference between baseline and current surface."""

    module: str
    name: str
    kind: str  # "added" | "removed" | "kind_changed" | "signature_changed"
    breaking: bool
    detail: str

    @property
    def location(self) -> str:
        return f"{self.module}.{self.name}"


def _member_kind(obj: Any) -> str:
    """Classify a member into a coarse, stable kind label."""

    if inspect.isclass(obj):
        return "class"
    if inspect.isfunction(obj) or inspect.isbuiltin(obj) or inspect.ismethod(obj):
        return "function"
    if inspect.ismodule(obj):
        return "module"
    if callable(obj):
        return "callable"
    return "constant"


def _member_signature(obj: Any, kind: str) -> str | None:
    """Return a normalized signature string for callables, else ``None``.

    Default *values* are deliberately dropped and only the presence of a default
    is recorded (as ``=...``). A changed default value is not a source-breaking
    change, while adding or removing a parameter -- or making an optional
    parameter required -- is. Capturing only defaultedness keeps the baseline
    stable against harmless default tweaks while still catching real breaks.
    """

    if kind not in {"class", "function", "callable"}:
        return None
    try:
        signature = inspect.signature(obj)
    except (TypeError, ValueError):
        return None

    parts: list[str] = []
    for parameter in signature.parameters.values():
        rendered = parameter.name
        if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            rendered = f"*{parameter.name}"
        elif parameter.kind == inspect.Parameter.VAR_KEYWORD:
            rendered = f"**{parameter.name}"
        elif parameter.kind == inspect.Parameter.KEYWORD_ONLY:
            rendered = f"kw:{parameter.name}"
        elif parameter.kind == inspect.Parameter.POSITIONAL_ONLY:
            rendered = f"po:{parameter.name}"
        if parameter.default is not inspect.Parameter.empty:
            rendered += "=..."
        parts.append(rendered)
    return "(" + ", ".join(parts) + ")"


def capture_module(module_name: str) -> SurfaceSnapshot | None:
    """Capture the public members of ``module_name`` via its ``__all__``.

    Returns ``None`` when the module cannot be imported (an optional backend is
    missing, for example) so that environments without every extra installed do
    not report spurious removals.
    """

    try:
        module = importlib.import_module(module_name)
    except Exception:  # pragma: no cover - depends on optional extras
        return None

    exported = getattr(module, "__all__", None)
    if exported is None:
        return SurfaceSnapshot(module=module_name, members=())

    members: list[MemberSnapshot] = []
    for name in sorted(dict.fromkeys(exported)):
        if not hasattr(module, name):
            # Declared in ``__all__`` but not importable; skip rather than fail
            # so the tool never masks a real removal with a capture error.
            continue
        obj = getattr(module, name)
        kind = _member_kind(obj)
        signature = _member_signature(obj, kind)
        members.append(MemberSnapshot(name=name, kind=kind, signature=signature))
    return SurfaceSnapshot(module=module_name, members=tuple(members))


def capture_surface(
    top_level: str = TOP_LEVEL_PACKAGE,
    subpackages: Sequence[str] = PUBLIC_SUBPACKAGES,
) -> list[SurfaceSnapshot]:
    """Capture the whole public surface as an ordered list of snapshots."""

    snapshots: list[SurfaceSnapshot] = []
    for module_name in [top_level, *subpackages]:
        snapshot = capture_module(module_name)
        if snapshot is not None:
            snapshots.append(snapshot)
    return snapshots


def surface_to_document(snapshots: Sequence[SurfaceSnapshot]) -> dict[str, Any]:
    """Serialize captured snapshots into a deterministic JSON-ready document."""

    modules: dict[str, Any] = {}
    for snapshot in sorted(snapshots, key=lambda item: item.module):
        modules[snapshot.module] = {
            member.name: {"kind": member.kind, "signature": member.signature}
            for member in sorted(snapshot.members, key=lambda item: item.name)
        }
    return {"schema_version": SCHEMA_VERSION, "modules": modules}


def document_to_surface(document: Mapping[str, Any]) -> list[SurfaceSnapshot]:
    """Rehydrate snapshots from a baseline document."""

    modules = document.get("modules", {})
    if not isinstance(modules, Mapping):
        raise ValueError("baseline document is missing a 'modules' table")

    snapshots: list[SurfaceSnapshot] = []
    for module_name in sorted(modules):
        raw_members = modules[module_name]
        if not isinstance(raw_members, Mapping):
            raise ValueError(f"baseline entry for {module_name!r} is malformed")
        members = tuple(
            MemberSnapshot(
                name=name,
                kind=str(raw_members[name].get("kind", "constant")),
                signature=raw_members[name].get("signature"),
            )
            for name in sorted(raw_members)
        )
        snapshots.append(SurfaceSnapshot(module=module_name, members=members))
    return snapshots


def _signature_is_breaking(old: str | None, new: str | None) -> bool:
    """Return whether a signature change is source-breaking.

    Adding a new parameter that has a default, or widening ``*args``/``**kwargs``
    acceptance, is not breaking for existing callers. Removing or renaming a
    parameter, reordering positional parameters, or turning an optional
    parameter into a required one is breaking.
    """

    if old is None or new is None:
        # A callable gaining or losing an introspectable signature is treated as
        # breaking; it usually signals a kind change already flagged elsewhere.
        return old != new

    old_params = _parse_signature(old)
    new_params = _parse_signature(new)
    new_by_name = dict(new_params)

    for name, had_default in old_params:
        if name not in new_by_name:
            return True  # removed or renamed parameter
        if had_default and not new_by_name[name]:
            return True  # optional parameter became required

    # A brand-new *required* parameter (not var-args) breaks existing callers.
    old_names = {name for name, _ in old_params}
    for name, has_default in new_params:
        if name in old_names:
            continue
        if name.startswith("*"):
            continue  # widening with *args/**kwargs is compatible
        if not has_default:
            return True

    # Reordering of shared positional parameters breaks positional callers.
    old_positional = [n for n, _ in old_params if not n.startswith("*")]
    new_positional = [n for n, _ in new_params if not n.startswith("*")]
    new_positional_set = set(new_positional)
    old_positional_set = set(old_positional)
    shared_old_order = [n for n in old_positional if n in new_positional_set]
    shared_new_order = [n for n in new_positional if n in old_positional_set]
    if shared_old_order != shared_new_order:
        return True

    return False


def _parse_signature(signature: str) -> list[tuple[str, bool]]:
    """Parse a captured signature string into ``(name, has_default)`` pairs."""

    inner = signature.strip()
    if inner.startswith("(") and inner.endswith(")"):
        inner = inner[1:-1]
    params: list[tuple[str, bool]] = []
    for chunk in inner.split(","):
        token = chunk.strip()
        if not token:
            continue
        has_default = token.endswith("=...")
        if has_default:
            token = token[: -len("=...")]
        # Strip the kind prefixes used during capture.
        for prefix in ("kw:", "po:"):
            if token.startswith(prefix):
                token = token[len(prefix) :]
        params.append((token, has_default))
    return params


def diff_surface(
    baseline: Sequence[SurfaceSnapshot],
    current: Sequence[SurfaceSnapshot],
) -> list[Change]:
    """Classify every difference between the baseline and current surfaces.

    Modules present in the baseline but absent from the current capture (because
    an optional dependency is missing) are skipped rather than reported as
    removed, so partial environments do not emit false breaks.
    """

    current_by_module = {snapshot.module: snapshot for snapshot in current}
    changes: list[Change] = []

    for baseline_snapshot in baseline:
        module = baseline_snapshot.module
        current_snapshot = current_by_module.get(module)
        if current_snapshot is None:
            continue  # module not importable here; do not infer a removal

        baseline_members = {m.name: m for m in baseline_snapshot.members}
        current_members = {m.name: m for m in current_snapshot.members}

        for name, old_member in baseline_members.items():
            new_member = current_members.get(name)
            if new_member is None:
                changes.append(
                    Change(
                        module=module,
                        name=name,
                        kind="removed",
                        breaking=True,
                        detail=f"public export {name!r} was removed",
                    )
                )
                continue
            if old_member.kind != new_member.kind:
                changes.append(
                    Change(
                        module=module,
                        name=name,
                        kind="kind_changed",
                        breaking=True,
                        detail=(
                            f"{name!r} changed kind: "
                            f"{old_member.kind} -> {new_member.kind}"
                        ),
                    )
                )
                continue
            if old_member.signature != new_member.signature:
                breaking = _signature_is_breaking(
                    old_member.signature, new_member.signature
                )
                changes.append(
                    Change(
                        module=module,
                        name=name,
                        kind="signature_changed",
                        breaking=breaking,
                        detail=(
                            f"{name!r} signature changed: "
                            f"{old_member.signature} -> {new_member.signature}"
                        ),
                    )
                )

        for name in current_members.keys() - baseline_members.keys():
            changes.append(
                Change(
                    module=module,
                    name=name,
                    kind="added",
                    breaking=False,
                    detail=f"new public export {name!r}",
                )
            )

    # A module that is entirely new (present now, absent from baseline) is an
    # addition of every one of its members.
    baseline_modules = {snapshot.module for snapshot in baseline}
    for current_snapshot in current:
        if current_snapshot.module in baseline_modules:
            continue
        for member in current_snapshot.members:
            changes.append(
                Change(
                    module=current_snapshot.module,
                    name=member.name,
                    kind="added",
                    breaking=False,
                    detail=f"new public export {member.name!r} in new module",
                )
            )

    return sorted(changes, key=lambda change: (change.module, change.name, change.kind))


def load_allowlist(path: Path) -> set[str]:
    """Load the set of intentionally-announced breaking-change locations.

    The allowlist file maps ``"module.name"`` -> free-form justification. Only
    the keys matter for gating; the values document why the break is sanctioned.
    """

    if not path.exists():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("announced_breaks", data) if isinstance(data, Mapping) else {}
    if not isinstance(entries, Mapping):
        raise ValueError("allowlist must map 'module.name' to a justification")
    return set(entries.keys())


def unannounced_breaks(
    changes: Iterable[Change],
    allowlist: set[str],
) -> list[Change]:
    """Return breaking changes that are not recorded in the allowlist."""

    return [
        change
        for change in changes
        if change.breaking and change.location not in allowlist
    ]


def load_baseline(path: Path) -> list[SurfaceSnapshot]:
    """Load and rehydrate the committed baseline document."""

    if not path.exists():
        raise FileNotFoundError(
            f"baseline not found at {path}; generate it with --update"
        )
    document = json.loads(path.read_text(encoding="utf-8"))
    return document_to_surface(document)


def write_baseline(path: Path, snapshots: Sequence[SurfaceSnapshot]) -> None:
    """Serialize the current surface to a deterministic baseline JSON file."""

    document = surface_to_document(snapshots)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def print_report(changes: Sequence[Change], failures: Sequence[Change]) -> None:
    """Print a compact, human-readable diff summary."""

    additions = [change for change in changes if change.kind == "added"]
    failure_set = set(failures)
    announced = [
        change for change in changes if change.breaking and change not in failure_set
    ]

    if failures:
        print("Public API check failed: unannounced breaking changes", file=sys.stderr)
        for change in failures:
            print(f"- BREAKING {change.location}: {change.detail}", file=sys.stderr)
        print(
            "\nAnnounce and allowlist these in "
            "scripts/release/public_api_allowlist.json, following "
            "docs/compliance/api-deprecation-policy.md, or revert the change.",
            file=sys.stderr,
        )
    else:
        print("Public API check passed")

    if announced:
        print(f"\nAllowlisted breaking changes ({len(announced)}):")
        for change in announced:
            print(f"- {change.location}: {change.detail}")
    if additions:
        print(f"\nBackwards-compatible additions ({len(additions)}):")
        for change in additions:
            print(f"- {change.location}: {change.detail}")


def run_check(
    baseline_path: Path = DEFAULT_BASELINE,
    allowlist_path: Path = DEFAULT_ALLOWLIST,
) -> int:
    """Run the check against the committed baseline and return an exit code."""

    baseline = load_baseline(baseline_path)
    current = capture_surface()
    changes = diff_surface(baseline, current)
    allowlist = load_allowlist(allowlist_path)
    failures = unannounced_breaks(changes, allowlist)
    print_report(changes, failures)
    return 1 if failures else 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Check the openmed public API surface against a committed baseline "
            "and fail on unannounced breaking changes."
        )
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE,
        help="Path to the committed public-API baseline JSON.",
    )
    parser.add_argument(
        "--allowlist",
        type=Path,
        default=DEFAULT_ALLOWLIST,
        help="Path to the announced-breaks allowlist JSON.",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Regenerate the baseline from the current surface and exit.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    args = parse_args(argv)
    if args.update:
        snapshots = capture_surface()
        write_baseline(args.baseline, snapshots)
        print(f"Wrote public API baseline to {args.baseline}")
        return 0
    return run_check(args.baseline, args.allowlist)


if __name__ == "__main__":
    raise SystemExit(main())
