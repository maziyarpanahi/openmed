#!/usr/bin/env python3
"""Dump, normalize and check the OpenMedKit Swift public API baselines.

For every OpenMedKit library product the script runs ``swift-api-digester
-dump-sdk`` against the modules of an existing root-package build and turns the
digester JSON into a small, toolchain-neutral baseline at
``swift/OpenMedKit/api/<platform>/<Module>.json``. Each baseline maps a
module-qualified symbol (``OpenMedKit.OpenMed.analyzeText``) to the
declarations recorded for it: labels, parameter and result types,
default-argument presence, ``throws`` and ``async`` effects, accessors,
conformances and generic signatures.

Two platforms are recorded because part of the surface is platform specific:
``macos`` (from ``swift build``) and ``ios`` (from an iOS Simulator
``xcodebuild`` build, which also covers the UIKit-only Share and Action
extension targets). Raw digester output is never committed: it embeds absolute
paths and fields that change between Xcode releases.

Typical use on macOS::

    swift build
    python3 scripts/release/swift_api_baseline.py --platform macos --update
    xcodebuild build -scheme OpenMedKit-Package \
        -destination 'generic/platform=iOS Simulator' \
        -derivedDataPath .build/xcode-ios -skipPackagePluginValidation
    python3 scripts/release/swift_api_baseline.py --platform ios --update

``--check`` (the default) fails when a recorded declaration was removed or
changed and names each symbol. New declarations never fail the check; it lists
the ones the baseline does not record yet so ``--update`` can add them.
Removals and changes also need a ``docs/migration/`` entry, enforced by
``check_native_api_migration.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import check_native_api_migration as native_api  # noqa: E402

ROOT = SCRIPT_DIR.parents[1]
BASELINE_DIR = ROOT / "swift" / "OpenMedKit" / "api"
SCHEMA_VERSION = native_api.SWIFT_SCHEMA_VERSION

# Library products declared by both Package.swift manifests.
MODULES: tuple[str, ...] = (
    "OpenMedKit",
    "OpenMedSpeech",
    "OpenMedExtensionSupport",
    "OpenMedShareExtension",
    "OpenMedActionExtension",
)


@dataclass(frozen=True)
class PlatformSpec:
    """How to locate a build and dump it for one Apple platform."""

    name: str
    sdk: str
    triple: str  # ``{arch}`` is replaced with the host architecture.
    build_hint: str
    # Candidate module directories, relative to the repository root. SwiftPM
    # 6 writes modules to ``.build/debug/Modules``; SwiftPM 5.10 does not.
    modules: tuple[str, ...]
    checkouts: str


_XCODE_IOS = ".build/xcode-ios"
# Deployment targets match ``platforms:`` in Package.swift.
PLATFORM_SPECS: dict[str, PlatformSpec] = {
    "macos": PlatformSpec(
        name="macos",
        sdk="macosx",
        triple="{arch}-apple-macosx14.0",
        build_hint="swift build",
        modules=(".build/debug/Modules", ".build/debug"),
        checkouts=".build/checkouts",
    ),
    "ios": PlatformSpec(
        name="ios",
        sdk="iphonesimulator",
        triple="{arch}-apple-ios17.0-simulator",
        build_hint=(
            "xcodebuild build -scheme OpenMedKit-Package -destination "
            f"'generic/platform=iOS Simulator' -derivedDataPath {_XCODE_IOS} "
            "-skipPackagePluginValidation"
        ),
        modules=(f"{_XCODE_IOS}/Build/Products/Debug-iphonesimulator",),
        checkouts=f"{_XCODE_IOS}/SourcePackages/checkouts",
    ),
}


def default_modules_dir(spec: PlatformSpec, module: str, root: Path = ROOT) -> Path:
    """Return the first build directory of ``spec`` that contains ``module``."""

    candidates = [root / candidate for candidate in spec.modules]
    for candidate in candidates:
        if (candidate / f"{module}.swiftmodule").exists():
            return candidate
    return candidates[0]


# Marker protocols that newer compilers attach implicitly to every type. They
# say nothing about the OpenMedKit contract and differ between Xcode releases.
IMPLICIT_CONFORMANCES = frozenset(
    {"Copyable", "Escapable", "BitwiseCopyable", "SendableMetatype"}
)
DECLARATION_KINDS = frozenset(
    {"TypeDecl", "Function", "Constructor", "Var", "Subscript", "TypeAlias"}
    | {"AssociatedType", "Macro"}
)
OWNERSHIP = {"InOut": "inout", "Owned": "__owned", "Shared": "__shared"}

Demangler = Callable[[Sequence[str]], Mapping[str, str]]


class SwiftApiError(RuntimeError):
    """Raised when the digester cannot produce a usable dump."""


def _tool(name: str) -> list[str]:
    if shutil.which("xcrun"):
        return ["xcrun", name]
    found = shutil.which(name)
    if found is None:
        raise SwiftApiError(f"{name} is not available; install a Swift toolchain")
    return [found]


def demangle(mangled_names: Sequence[str]) -> dict[str, str]:
    """Demangle Swift symbol names with the toolchain's ``swift-demangle``."""

    if not mangled_names:
        return {}
    result = subprocess.run(
        [*_tool("swift-demangle"), "--compact"],
        input="\n".join(mangled_names) + "\n",
        capture_output=True,
        check=False,
        text=True,
    )
    lines = result.stdout.splitlines()
    if result.returncode != 0 or len(lines) != len(mangled_names):
        raise SwiftApiError("swift-demangle failed")
    return dict(zip(mangled_names, lines))


def _effects_after_parameters(demangled: str) -> str:
    """Return the effect text after a demangled declaration's parameter list."""

    start = None
    angle = 0
    for index, char in enumerate(demangled):
        if char == "<":
            angle += 1
        elif char == ">" and angle and demangled[index - 1] != "-":
            angle -= 1
        elif (
            char == "("
            and angle == 0
            and not demangled.startswith("(extension in", index)
        ):
            start = index
            break
    if start is None:
        return ""
    depth = 0
    for index in range(start, len(demangled)):
        if demangled[index] == "(":
            depth += 1
        elif demangled[index] == ")":
            depth -= 1
            if depth == 0:
                return demangled[index + 1 :]
    return ""


def is_async(demangled: str | None) -> bool:
    """Return whether a demangled function or initializer is ``async``.

    ``swift-api-digester`` API dumps omit the async effect, so it is read from
    the declaration's own mangled name rather than from closure parameters.
    """

    if not demangled:
        return False
    return _effects_after_parameters(demangled).lstrip().startswith("async")


def _labels(printed_name: str) -> list[str]:
    if "(" not in printed_name or not printed_name.endswith(")"):
        return []
    inner = printed_name[printed_name.index("(") + 1 : -1]
    return [label for label in inner.split(":") if label]


def _base_name(printed_name: str) -> str:
    return printed_name.split("(", 1)[0]


def _type_name(node: Mapping[str, Any]) -> str:
    return str(node.get("printedName", ""))


def _parameters(node: Mapping[str, Any]) -> str:
    children = list(node.get("children", []))[1:]
    labels = _labels(str(node.get("printedName", "")))
    rendered = []
    for index, child in enumerate(children):
        label = labels[index] if index < len(labels) else "_"
        ownership = OWNERSHIP.get(str(child.get("paramValueOwnership", "")), "")
        if not ownership and child.get("paramValueOwnership"):
            ownership = str(child["paramValueOwnership"]).lower()
        type_text = f"{ownership} {_type_name(child)}".strip()
        default = " = default" if child.get("hasDefaultArg") else ""
        rendered.append(f"{label}: {type_text}{default}")
    return "(" + ", ".join(rendered) + ")"


def _accessors(node: Mapping[str, Any]) -> str:
    kinds = []
    for accessor in node.get("accessors", []):
        kind = str(accessor.get("accessorKind", "")).lstrip("_")
        if kind in {"get", "set"} and kind not in kinds:
            kinds.append(kind)
        elif kind == "modify" and "set" not in kinds:
            kinds.append("set")
    order = {"get": 0, "set": 1}
    return "{ " + " ".join(sorted(kinds, key=order.__getitem__)) + " }"


def _prefix(node: Mapping[str, Any]) -> str:
    words = []
    if node.get("static"):
        words.append("static")
    if node.get("funcSelfKind") == "Mutating":
        words.append("mutating")
    if node.get("init_kind") == "Convenience":
        words.append("convenience")
    attributes = set(node.get("declAttributes", []))
    if "Nonisolated" in attributes:
        words.append("nonisolated")
    if "Preconcurrency" in attributes:
        words.append("@preconcurrency")
    return " ".join(words) + (" " if words else "")


def render_declarations(
    node: Mapping[str, Any],
    qualified: str,
    demangled: Mapping[str, str],
) -> list[str]:
    """Render the baseline lines for one digester declaration node.

    Args:
        node: A digester ``TypeDecl``, ``Function``, ``Var`` or similar node.
        qualified: Module-qualified printed name, labels included.
        demangled: Demangled names keyed by digester ``mangledName``.

    Returns:
        Declaration lines; a type contributes one line per conformance.
    """

    kind = node.get("kind")
    decl_kind = str(node.get("declKind", ""))
    generic = str(node.get("genericSig", ""))
    prefix = _prefix(node)
    if kind == "TypeDecl":
        attributes = set(node.get("declAttributes", []))
        keyword = decl_kind.lower()
        if decl_kind == "Class" and "Final" in attributes:
            keyword = "final class"
        line = f"{prefix}{keyword} {qualified}{generic}"
        if node.get("enumRawTypeName"):
            line += f" raw({node['enumRawTypeName']})"
        lines = [line]
        for conformance in node.get("conformances", []):
            name = str(conformance.get("printedName", ""))
            if name and name not in IMPLICIT_CONFORMANCES:
                lines.append(f"conformance {qualified}: {name}")
        return lines
    children = list(node.get("children", []))
    result_type = _type_name(children[0]) if children else "()"
    if kind in {"Function", "Constructor", "Subscript"}:
        # Initializers and subscripts already carry their keyword in the name.
        keyword = "func " if kind == "Function" else ""
        name = _base_name(qualified) + generic
        effects = []
        if is_async(demangled.get(str(node.get("mangledName", "")))):
            effects.append("async")
        if node.get("throwing"):
            effects.append("throws")
        effect_text = "".join(f" {effect}" for effect in effects)
        line = f"{prefix}{keyword}{name}{_parameters(node)}{effect_text}"
        line += f" -> {result_type}"
        if kind == "Subscript":
            line += f" {_accessors(node)}"
        return [line]
    if decl_kind == "EnumElement":
        payload = ""
        if children and children[0].get("kind") == "TypeFunc":
            inner = list(children[0].get("children", []))
            if inner and inner[0].get("kind") == "TypeFunc":
                payload = _type_name(inner[0])
        return [f"case {qualified}" + (f": {payload}" if payload else "")]
    if kind == "Var":
        return [f"{prefix}var {qualified}: {result_type} {_accessors(node)}"]
    if kind == "TypeAlias":
        return [f"typealias {qualified}{generic} = {result_type}"]
    return [f"{prefix}{decl_kind.lower() or str(kind).lower()} {qualified}{generic}"]


def _symbol_key(qualified: str) -> str:
    return _base_name(qualified)


def _walk(
    node: Mapping[str, Any],
    parents: tuple[str, ...],
    *,
    top_level: bool = False,
) -> Iterable[tuple[Mapping[str, Any], str]]:
    for child in node.get("children", []):
        if child.get("kind") not in DECLARATION_KINDS or child.get("implicit"):
            continue
        name = str(child.get("printedName") or child.get("name") or "")
        if not name:
            continue
        if top_level and child.get("kind") == "TypeDecl" and child.get("moduleName"):
            # Extensions of another module's type nest under that module.
            path = (str(child["moduleName"]), name)
        else:
            path = (*parents, name)
        yield child, ".".join(path)
        if child.get("kind") == "TypeDecl":
            yield from _walk(child, path)


def _mangled_names(root: Mapping[str, Any]) -> list[str]:
    names: list[str] = []

    def visit(node: Mapping[str, Any]) -> None:
        if node.get("kind") in {"Function", "Constructor", "Subscript"}:
            mangled = node.get("mangledName")
            if isinstance(mangled, str) and mangled:
                names.append(mangled)
        for child in node.get("children", []):
            if child.get("kind") in DECLARATION_KINDS:
                visit(child)

    visit(root)
    return sorted(set(names))


def normalize_dump(
    dump: Mapping[str, Any],
    module: str,
    demangler: Demangler | None = None,
) -> dict[str, Any]:
    """Turn ``swift-api-digester -dump-sdk`` JSON into a baseline document.

    Args:
        dump: Parsed digester JSON (with its ``ABIRoot`` object).
        module: Module name the dump was produced for.
        demangler: Callable that demangles a batch of mangled names;
            defaults to the toolchain's ``swift-demangle``.

    Returns:
        A JSON-serializable baseline with sorted symbols and declarations.

    Raises:
        SwiftApiError: If the dump does not describe ``module``.
    """

    root = dump.get("ABIRoot")
    if not isinstance(root, dict) or root.get("name") != module:
        raise SwiftApiError(f"digester dump does not describe module {module}")
    demangled = (demangler or demangle)(_mangled_names(root))
    table: dict[str, set[str]] = {}
    for node, path in _walk(root, (module,), top_level=True):
        node_module = node.get("moduleName")
        if node_module and node_module != module:
            continue
        for line in render_declarations(node, path, demangled):
            table.setdefault(_symbol_key(path), set()).add(line)
    return {
        "schema_version": SCHEMA_VERSION,
        "module": module,
        "generator": "swift-api-digester -dump-sdk",
        "symbols": {symbol: sorted(table[symbol]) for symbol in sorted(table)},
    }


def render_baseline(document: Mapping[str, Any]) -> str:
    """Serialize a baseline document deterministically."""

    return json.dumps(document, indent=2, ensure_ascii=False) + "\n"


def _clang_search_flags(checkouts: Path) -> list[str]:
    """Return ``-I`` flags for the C modules of resolved package checkouts."""

    flags: list[str] = []
    if checkouts.is_dir():
        for module_map in sorted(checkouts.glob("*/Source*/**/module.modulemap")):
            relative_parts = module_map.relative_to(checkouts).parts
            if any(
                part in {"Tests", "Benchmarks", "Examples"} for part in relative_parts
            ):
                continue
            flags.extend(["-I", str(module_map.parent)])
    return flags


def _host_arch() -> str:
    return "arm64" if platform.machine() in {"arm64", "aarch64"} else "x86_64"


def run_digester(
    module: str,
    spec: PlatformSpec,
    modules_dir: Path,
    checkouts: Path,
    output: Path,
) -> dict[str, Any]:
    """Dump ``module`` from an existing build with ``swift-api-digester``.

    Args:
        module: Swift module name.
        spec: Platform whose SDK and target triple to use.
        modules_dir: Directory holding the built ``<Module>.swiftmodule``.
        checkouts: SwiftPM checkouts directory, for C module maps.
        output: Where the raw digester JSON is written.

    Returns:
        The parsed digester JSON.

    Raises:
        SwiftApiError: If the module was not built or the digester fails.
    """

    if not (modules_dir / f"{module}.swiftmodule").exists():
        raise SwiftApiError(
            f"{module}.swiftmodule was not found for {spec.name}; "
            f"build first with `{spec.build_hint}`"
        )
    sdk = subprocess.run(
        ["xcrun", "--sdk", spec.sdk, "--show-sdk-path"],
        capture_output=True,
        check=False,
        text=True,
    )
    if sdk.returncode != 0:
        raise SwiftApiError(f"could not locate the {spec.sdk} SDK with xcrun")
    command = [
        *_tool("swift-api-digester"),
        "-dump-sdk",
        "-module",
        module,
        "-o",
        str(output),
        "-sdk",
        sdk.stdout.strip(),
        "-target",
        spec.triple.format(arch=_host_arch()),
        "-I",
        str(modules_dir),
        *_clang_search_flags(checkouts),
        "-abort-on-module-fail",
        "-avoid-location",
    ]
    result = subprocess.run(command, capture_output=True, check=False, text=True)
    if result.returncode != 0 or not output.is_file():
        raise SwiftApiError(
            f"swift-api-digester failed for {module} on {spec.name} "
            f"(exit {result.returncode})"
        )
    return json.loads(output.read_text(encoding="utf-8"))


def current_baselines(
    modules: Sequence[str],
    *,
    spec: PlatformSpec = PLATFORM_SPECS["macos"],
    modules_dir: Path | None = None,
    checkouts: Path | None = None,
    digester_json: Path | None = None,
    demangler: Demangler | None = None,
) -> dict[str, dict[str, Any]]:
    """Return freshly normalized baselines for ``modules``.

    Args:
        modules: Swift modules to dump.
        spec: Platform being dumped.
        modules_dir: Directory holding built ``.swiftmodule`` files; defaults
            to the platform's standard build location.
        checkouts: SwiftPM checkouts directory; defaults per platform.
        digester_json: Optional directory of existing ``<Module>.json`` digester
            dumps to normalize instead of running the digester.
        demangler: Callable that demangles a batch of mangled names;
            defaults to the toolchain's ``swift-demangle``.

    Returns:
        Baseline documents keyed by module name.
    """

    documents: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="openmed-swift-api-") as temp:
        for module in modules:
            if digester_json is not None:
                raw_path = digester_json / f"{module}.json"
                try:
                    raw = json.loads(raw_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise SwiftApiError(
                        f"could not read digester dump for {module}"
                    ) from exc
            else:
                raw = run_digester(
                    module,
                    spec,
                    modules_dir or default_modules_dir(spec, module),
                    checkouts or ROOT / spec.checkouts,
                    Path(temp) / f"{module}.json",
                )
            documents[module] = normalize_dump(raw, module, demangler)
    return documents


def compare_baselines(
    committed: Mapping[str, str | None],
    current: Mapping[str, Mapping[str, Any]],
    platform_name: str = "macos",
) -> list[native_api.BaselineChange]:
    """Diff committed baseline text against freshly generated documents.

    Args:
        committed: Baseline text keyed by module; ``None`` when missing.
        current: Freshly normalized documents keyed by module.
        platform_name: Platform directory the baselines belong to.

    Returns:
        Changes from the committed baseline to the current surface.
    """

    changes: list[native_api.BaselineChange] = []
    for module, document in current.items():
        text = committed.get(module)
        before = native_api.parse_swift_baseline(text) if text is not None else {}
        after = native_api.parse_swift_baseline(render_baseline(document))
        changes.extend(
            native_api.diff_tables(
                before,
                after,
                platform="swift",
                baseline=f"swift/OpenMedKit/api/{platform_name}/{module}.json",
            )
        )
    return changes


def _read_committed(
    baseline_dir: Path, modules: Sequence[str]
) -> dict[str, str | None]:
    committed: dict[str, str | None] = {}
    for module in modules:
        path = baseline_dir / f"{module}.json"
        committed[module] = path.read_text(encoding="utf-8") if path.is_file() else None
    return committed


def run_check(
    current: Mapping[str, Mapping[str, Any]],
    baseline_dir: Path,
    platform_name: str = "macos",
) -> int:
    """Compare ``current`` with the committed baselines; return an exit code."""

    committed = _read_committed(baseline_dir, list(current))
    changes = compare_baselines(committed, current, platform_name)
    breaking = [change for change in changes if change.breaking]
    added = [change for change in changes if not change.breaking]
    update = (
        "python3 scripts/release/swift_api_baseline.py "
        f"--platform {platform_name} --update"
    )
    if not changes:
        print(
            f"Swift API baselines match for {len(current)} module(s) "
            f"on {platform_name}."
        )
        return 0
    print(native_api.render_report(changes))
    if added:
        print(
            f"\n{len(added)} public Swift symbol(s) are not in the {platform_name} "
            f"baseline yet. Additions do not fail this check; record them with "
            f"`{update}` (no migration entry needed):"
        )
        for change in added:
            print(f"  + {change.symbol}")
    if not breaking:
        return 0
    sys.stdout.flush()
    print(
        f"\n{len(breaking)} public Swift symbol(s) were removed or changed on "
        f"{platform_name}. If intended, refresh the baselines with `{update}` and "
        "name each symbol in a docs/migration/ entry. If only the Xcode toolchain "
        "changed, refresh the baselines in a separate change.",
        file=sys.stderr,
    )
    if os.environ.get("GITHUB_ACTIONS") == "true":
        for change in breaking:
            print(
                f"::error title=Swift API {change.change}::{change.symbol} "
                f"({platform_name})"
            )
    return 1


def write_baselines(
    documents: Mapping[str, Mapping[str, Any]], destination: Path
) -> None:
    """Write normalized baseline documents to ``destination``."""

    destination.mkdir(parents=True, exist_ok=True)
    for module, document in documents.items():
        (destination / f"{module}.json").write_text(
            render_baseline(document), encoding="utf-8"
        )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check",
        action="store_true",
        help="Compare the current API with the committed baselines (default)",
    )
    mode.add_argument(
        "--update",
        action="store_true",
        help="Rewrite the committed baselines from the current build",
    )
    parser.add_argument(
        "--module",
        action="append",
        choices=MODULES,
        dest="modules",
        help="Limit to one module (repeatable; default: every library product)",
    )
    parser.add_argument(
        "--platform",
        choices=sorted(PLATFORM_SPECS),
        default="macos",
        help="Platform build to dump (default: macos)",
    )
    parser.add_argument(
        "--modules-dir",
        type=Path,
        help="Directory with the built .swiftmodule files (default per platform)",
    )
    parser.add_argument(
        "--checkouts",
        type=Path,
        help="SwiftPM checkouts directory for C module maps (default per platform)",
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        help="Committed baseline directory (default swift/OpenMedKit/api/<platform>)",
    )
    parser.add_argument(
        "--digester-json",
        type=Path,
        help="Normalize existing <Module>.json digester dumps from this directory",
    )
    parser.add_argument(
        "--write-current",
        type=Path,
        metavar="DIR",
        help="Also write the freshly normalized baselines to DIR",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Swift API baseline tool from the command line."""

    args = build_parser().parse_args(argv)
    modules = tuple(args.modules or MODULES)
    spec = PLATFORM_SPECS[args.platform]
    baseline_dir = args.baseline_dir or BASELINE_DIR / spec.name
    try:
        current = current_baselines(
            modules,
            spec=spec,
            modules_dir=args.modules_dir,
            checkouts=args.checkouts,
            digester_json=args.digester_json,
        )
        if args.write_current is not None:
            write_baselines(current, args.write_current)
        if args.update:
            write_baselines(current, baseline_dir)
            print(
                f"Updated Swift API baselines for {len(current)} module(s) "
                f"on {spec.name}."
            )
            return 0
        return run_check(current, baseline_dir, spec.name)
    except (SwiftApiError, native_api.NativeApiError) as exc:
        print(f"swift-api-baseline: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
