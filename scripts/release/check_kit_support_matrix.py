#!/usr/bin/env python3
"""Check the frozen OpenMedKit support matrix against the repository.

The matrix data lives in ``openmed/eval/data/kit_support_matrix.json``. This
check fails when a package manifest pin, a platform floor, the export family
list or the tier table changes without a matching matrix update, when runtime
skew between kits is not listed explicitly, when a supported cell has no
evidence, or when ``docs/openmedkit-support-matrix.md`` no longer matches the
rendered matrix.

Usage::

    python scripts/release/check_kit_support_matrix.py
    python scripts/release/check_kit_support_matrix.py --write-docs

Output lists fixed problem codes plus controlled matrix identifiers. It reads
repository metadata only and makes no network calls.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import sys
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

# A direct ``python scripts/release/...`` invocation otherwise puts only
# ``scripts/release`` first on ``sys.path``; pin imports to this checkout.
_root_string = str(ROOT)
if _root_string in sys.path:
    sys.path.remove(_root_string)
sys.path.insert(0, _root_string)

from openmed.eval.kit_support_matrix import (  # noqa: E402
    ANY,
    DEFAULT_KIT_SUPPORT_MATRIX,
    DOC_BEGIN_MARKER,
    DOC_END_MARKER,
    UNSUPPORTED,
    iter_entries,
    load_kit_support_matrix,
    render_kit_support_matrix_markdown,
    validate_kit_support_matrix,
    version_key,
)
from openmed.eval.tiers import TIERS  # noqa: E402

DOC_PATH = Path("docs") / "openmedkit-support-matrix.md"
EXPORT_MATRIX_DOC = "docs/export-matrix.md"
TIERS_SOURCE = "openmed.eval.tiers.TIERS"
MKDOCS = Path("mkdocs.yml")

# Manifests that may pin ONNX Runtime. Any of them carrying a version literal
# must be the source of an ``onnxruntime`` pin in the matrix.
ROOT_MANIFESTS = (
    "Package.swift",
    "swift/OpenMedKit/Package.swift",
    "pyproject.toml",
    "android/gradle/libs.versions.toml",
)
MANIFEST_TREES = ("android", "js")
MANIFEST_NAMES = (
    "build.gradle",
    "build.gradle.kts",
    "package.json",
    "*.podspec",
    "CMakeLists.txt",
    "Cargo.toml",
)
_SKIPPED_DIRECTORIES = frozenset({"node_modules", "build", ".gradle", "target"})
_ORT_PIN_LITERAL = re.compile(
    r"onnx_?runtime[\w.:-]*(?:\s*(?:[<>=~!]=?|\^)|['\"]?\s*[,:=]\s*['\"]?[<>=~^]*)"
    r"\s*v?\d+\.\d+",
    re.IGNORECASE,
)

# Export-matrix column that certifies an artifact format at a precision.
EXPORT_COLUMNS: Mapping[tuple[str, str], str] = {
    ("mlx", "fp32"): "MLX fp",
    ("mlx", "fp16"): "MLX fp",
    ("mlx", "int8"): "MLX INT8",
    ("mlx", "int4"): "MLX INT4",
    ("coreml", "fp16"): "CoreML fp16",
    ("coreml", "int8"): "CoreML INT8",
    ("onnx", "fp32"): "ONNX (default)",
    ("onnx-webgpu", "fp16"): "ONNX (default)",
    ("onnx-android", "fp32"): "ONNX Android",
    ("onnx-android", "fp16"): "ONNX Android",
    ("onnx-android", "int8"): "ONNX Android",
    ("transformersjs", "fp32"): "Transformers.js",
    ("transformersjs", "int8"): "Transformers.js",
}
# Checkpoints loaded natively need no export column.
NATIVE_ARTIFACTS = frozenset({"pytorch"})

_SWIFT_PLATFORM_NAMES = {
    "iOS": "ios",
    "macOS": "macos",
    "watchOS": "watchos",
    "visionOS": "visionos",
    "tvOS": "tvos",
    "macCatalyst": "mac-catalyst",
}


class ManifestError(ValueError):
    """A manifest could not be read or did not contain the expected value."""


def _read(root: Path, relative: str) -> str:
    try:
        return (root / relative).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        raise ManifestError("manifest_unreadable") from None


def _search(pattern: str, text: str, flags: int = 0) -> str:
    match = re.search(pattern, text, flags)
    if match is None:
        raise ManifestError("manifest_value_missing")
    return match.group(1)


def swift_platforms(text: str) -> dict[str, str]:
    """Return the deployment floors declared in a ``Package.swift``.

    Args:
        text: Manifest source.

    Returns:
        Platform name to major version, plus ``swift-tools``.
    """
    block = _search(r"platforms:\s*\[(.*?)\]", text, re.DOTALL)
    platforms = {
        _SWIFT_PLATFORM_NAMES.get(name, name.lower()): major
        + (f".{minor}" if minor else "")
        for name, major, minor in re.findall(
            r"\.(\w+)\(\s*\.v(\d+)(?:_(\d+))?\s*\)", block
        )
    }
    platforms["swift-tools"] = _search(r"swift-tools-version:\s*([\d.]+)", text)
    return platforms


def swift_mlx_platforms(text: str) -> frozenset[str]:
    """Return the platforms the ``MLX`` product is linked for.

    Args:
        text: ``Package.swift`` source.

    Returns:
        Lower-case platform names from the product's ``.when`` condition.
    """
    block = _search(
        r'name:\s*"MLX",\s*package:\s*"mlx-swift",\s*'
        r"condition:\s*\.when\(platforms:\s*\[([^\]]*)\]\)",
        text,
    )
    return frozenset(
        _SWIFT_PLATFORM_NAMES.get(name, name.lower())
        for name in re.findall(r"\.(\w+)", block)
    )


def _swift_mlx_pin(text: str) -> str:
    return _search(r'mlx-swift\.git"\s*,\s*exact:\s*"([^"]+)"', text)


def _pyproject_extra_spec(project: Mapping[str, Any], extra: str, name: str) -> str:
    extras = project.get("optional-dependencies", {})
    specs = [
        requirement.split(";", 1)[0].strip()
        for requirement in extras.get(extra, [])
        if re.match(rf"{re.escape(name)}\s*[<>=~!]", requirement)
    ]
    if len(specs) != 1:
        raise ManifestError("manifest_value_missing")
    return specs[0][len(name) :].replace(" ", "")


def _pyproject(root: Path) -> Mapping[str, Any]:
    try:
        return tomllib.loads(_read(root, "pyproject.toml"))["project"]
    except (tomllib.TOMLDecodeError, KeyError):
        raise ManifestError("manifest_unreadable") from None


def _package_json(root: Path, relative: str) -> Mapping[str, Any]:
    try:
        document = json.loads(_read(root, relative))
    except json.JSONDecodeError:
        raise ManifestError("manifest_unreadable") from None
    if not isinstance(document, dict):
        raise ManifestError("manifest_unreadable")
    return document


def _json_field(document: Mapping[str, Any], section: str, name: str) -> str:
    value = document.get(section, {}).get(name)
    if not isinstance(value, str):
        raise ManifestError("manifest_value_missing")
    return value.replace(" ", "")


def floor_value(root: Path, source: str, floor: Mapping[str, Any]) -> str:
    """Read the version a manifest declares for one platform floor.

    Args:
        root: Repository root.
        source: Repository-relative manifest path named by the floor.
        floor: Floor entry from the matrix.

    Returns:
        The declared version, without a ``>=`` prefix.

    Raises:
        ManifestError: If the manifest or value is missing.
    """
    platform = floor["platform"]
    bound = floor["bound"]
    if source.endswith("Package.swift"):
        platforms = swift_platforms(_read(root, source))
        if platform not in platforms:
            raise ManifestError("manifest_value_missing")
        return platforms[platform]
    if source == "android/openmedkit/build.gradle.kts":
        key = {"min": "minSdk", "compile": "compileSdk", "target": "targetSdk"}[bound]
        return _search(rf"\b{key}\s*=\s*(\d+)", _read(root, source))
    if source == "js/openmedkit-flutter/android/build.gradle":
        key = {"min": "minSdk", "compile": "compileSdk"}.get(bound)
        if key is None:
            raise ManifestError("manifest_value_missing")
        return _search(rf"\b{key}\s+(\d+)", _read(root, source))
    if source.endswith(".podspec"):
        system = {"ios": "ios", "macos": "osx"}.get(platform)
        if system is None:
            raise ManifestError("manifest_value_missing")
        return _search(
            rf"s\.platform\s*=\s*:{system},\s*'([^']+)'", _read(root, source)
        )
    if source == "pyproject.toml":
        return _pyproject(root).get("requires-python", "").replace(" ", "")
    if source.endswith("Cargo.toml"):
        try:
            cargo = tomllib.loads(_read(root, source))
        except tomllib.TOMLDecodeError:
            raise ManifestError("manifest_unreadable") from None
        value = cargo.get("dependencies", {}).get(platform)
        if isinstance(value, Mapping):
            value = value.get("version")
        if not isinstance(value, str):
            raise ManifestError("manifest_value_missing")
        return value
    if source.endswith("package.json"):
        document = _package_json(root, source)
        if platform == "node":
            return _json_field(document, "engines", "node")
        name = {"tauri": "@tauri-apps/api"}.get(platform, platform)
        return _json_field(document, "peerDependencies", name)
    raise ManifestError("manifest_unknown")


def pin_value(root: Path, source: str, pin: Mapping[str, Any]) -> str:
    """Read the version a manifest declares for one runtime pin.

    Args:
        root: Repository root.
        source: Repository-relative manifest path named by the pin.
        pin: Runtime pin entry from the matrix.

    Returns:
        The declared version or constraint, without whitespace.

    Raises:
        ManifestError: If the manifest or value is missing.
    """
    runtime = pin["runtime"]
    if source.endswith("Package.swift") and runtime == "mlx-swift":
        return _swift_mlx_pin(_read(root, source))
    if source == "android/gradle/libs.versions.toml":
        try:
            catalog = tomllib.loads(_read(root, source))
        except tomllib.TOMLDecodeError:
            raise ManifestError("manifest_unreadable") from None
        value = catalog.get("versions", {}).get(runtime)
        if not isinstance(value, str):
            raise ManifestError("manifest_value_missing")
        return value
    if source == "js/openmedkit-flutter/android/build.gradle":
        return _search(r"onnxRuntimeVersion\s*=\s*'([^']+)'", _read(root, source))
    if source.endswith(".podspec"):
        return _search(
            r"s\.dependency\s+'onnxruntime-c',\s*'([^']+)'", _read(root, source)
        )
    if source == "js/openmedkit-flutter/native/CMakeLists.txt":
        text = _read(root, source)
        if "OPENMED_ONNXRUNTIME_ROOT" not in text:
            raise ManifestError("manifest_value_missing")
        return "unpinned" if _ORT_PIN_LITERAL.search(text) is None else "pinned"
    if source == "js/openmedkit-web/package.json":
        name = {
            "onnxruntime": "onnxruntime-web",
            "transformers-js": "@huggingface/transformers",
        }.get(runtime)
        if name is None:
            raise ManifestError("manifest_value_missing")
        return _json_field(_package_json(root, source), "peerDependencies", name)
    if source == "pyproject.toml":
        project = _pyproject(root)
        if runtime == "torch":
            return _pyproject_extra_spec(project, "hf", "torch")
        if runtime == "mlx":
            return _pyproject_extra_spec(project, "mlx", "mlx")
        if runtime == "onnxruntime":
            specs = {
                _pyproject_extra_spec(project, extra, "onnxruntime")
                for extra in ("onnx", "onnx-runtime", "edge-sbc")
            }
            if len(specs) != 1:
                raise ManifestError("manifest_value_mismatch")
            return specs.pop()
    raise ManifestError("manifest_unknown")


def parse_export_matrix(text: str) -> dict[str, dict[str, str]]:
    """Parse the family table in ``docs/export-matrix.md``.

    Args:
        text: Markdown source of the export matrix page.

    Returns:
        Family row name to column name to ``yes``, ``partial`` or ``no``.

    Raises:
        ManifestError: If the table header is missing.
    """
    lines = [line.strip() for line in text.splitlines()]
    try:
        start = next(i for i, line in enumerate(lines) if line.startswith("| Family |"))
    except StopIteration:
        raise ManifestError("export_matrix_table_missing") from None
    header = [cell.strip() for cell in lines[start].strip("|").split("|")]
    rows: dict[str, dict[str, str]] = {}
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        row = {}
        for column, value in zip(header[1:], cells[1:]):
            if value.startswith("✅"):
                row[column] = "yes"
            elif value.startswith("⚠"):
                row[column] = "partial"
            elif value.startswith("❌"):
                row[column] = "no"
            else:
                row[column] = "n/a"
        rows[cells[0]] = row
    return rows


def manifest_paths(root: Path) -> list[str]:
    """Return the repository-relative manifests scanned for runtime pins.

    Build output, dependency caches and ``node_modules`` are pruned.

    Args:
        root: Repository root.

    Returns:
        Sorted POSIX paths of every existing manifest the check reads.
    """
    found = {path for path in ROOT_MANIFESTS if (root / path).is_file()}
    for top in MANIFEST_TREES:
        base = root / top
        if not base.is_dir():
            continue
        for directory, subdirectories, files in os.walk(base):
            subdirectories[:] = sorted(
                name for name in subdirectories if name not in _SKIPPED_DIRECTORIES
            )
            for name in files:
                if any(fnmatch.fnmatch(name, pattern) for pattern in MANIFEST_NAMES):
                    found.add((Path(directory) / name).relative_to(root).as_posix())
    return sorted(found)


def check_repository_coherence(
    matrix: Mapping[str, Any],
    root: Path = ROOT,
    *,
    tiers: Mapping[str, Mapping[str, Any]] = TIERS,
    check_docs: bool = True,
) -> list[str]:
    """Cross-check the matrix against manifests, docs, tiers and evidence paths.

    Args:
        matrix: A decoded matrix document.
        root: Repository root to read manifests from.
        tiers: Tier table the matrix must resolve against.
        check_docs: Also require the rendered docs block and nav entry.

    Returns:
        Sorted, de-duplicated problem codes. Empty means coherent.
    """
    problems = set(validate_kit_support_matrix(matrix, tiers=tiers))
    if problems:
        # Repository checks assume a structurally valid document.
        return sorted(problems)

    sources = matrix["sources"]
    if sources.get("families") != EXPORT_MATRIX_DOC:
        problems.add("families_source_invalid")
    if sources.get("tiers") != TIERS_SOURCE:
        problems.add("tiers_source_invalid")

    _check_families(problems, matrix, root)
    _check_floors(problems, matrix, root)
    _check_mlx_linkage(problems, matrix, root)
    _check_pins(problems, matrix, root)
    _check_skew(problems, matrix, root)
    for record in matrix["evidence"]:
        if not all((root / ref).exists() for ref in record["refs"]):
            problems.add(f"evidence_ref_missing:{record['id']}")
    if check_docs:
        _check_docs(problems, matrix, root, tiers)
    return sorted(problems)


def _check_families(problems: set[str], matrix: Mapping[str, Any], root: Path) -> None:
    try:
        export = parse_export_matrix(_read(root, EXPORT_MATRIX_DOC))
    except ManifestError as exc:
        problems.add(str(exc))
        return
    rows = {family["id"]: family["export_matrix_row"] for family in matrix["families"]}
    if set(rows.values()) != set(export):
        problems.add("families_mismatch_export_matrix")
        return
    for entry in iter_entries(matrix):
        artifact = entry.get("artifact")
        families = entry["families"]
        if artifact is None or artifact in NATIVE_ARTIFACTS or families in ([], [ANY]):
            continue
        if entry["status"] == UNSUPPORTED:
            if entry["reason"] != "family-not-exported":
                continue
            columns = {
                column
                for (kind, precision), column in EXPORT_COLUMNS.items()
                if kind == artifact
                and (entry["precisions"] == [ANY] or precision in entry["precisions"])
            }
            if not columns or any(
                export[rows[family]].get(column) != "no"
                for family in families
                for column in columns
            ):
                problems.add(f"unsupported_family_is_exported:{entry['id']}")
            continue
        for precision in entry["precisions"]:
            column = EXPORT_COLUMNS.get((artifact, precision))
            if column is None:
                problems.add(f"cell_precision_not_exported:{entry['id']}")
                continue
            if any(export[rows[family]].get(column) == "no" for family in families):
                problems.add(f"cell_family_not_exported:{entry['id']}")
    for entry in matrix["unsupported"]:
        if entry["reason"] == "family-not-exported" and entry.get("artifact") is None:
            problems.add(f"unsupported_family_artifact_missing:{entry['id']}")


def _check_floors(problems: set[str], matrix: Mapping[str, Any], root: Path) -> None:
    floors = matrix["platform_floors"]
    for floor in floors:
        for source in floor["sources"]:
            try:
                actual = floor_value(root, source, floor)
                matches = version_key(actual) == version_key(floor["version"])
            except (ManifestError, ValueError):
                problems.add(f"floor_unreadable:{floor['id']}")
                continue
            if not matches:
                problems.add(f"floor_mismatch:{floor['id']}")

    declared = {
        floor["platform"]
        for floor in floors
        if floor["surface"] == "swift"
        and floor["bound"] == "min"
        and floor["platform"] != "swift-tools"
    }
    for source in ("Package.swift", "swift/OpenMedKit/Package.swift"):
        try:
            platforms = set(swift_platforms(_read(root, source))) - {"swift-tools"}
        except ManifestError:
            problems.add("floor_platforms_unreadable:swift")
            continue
        if platforms != declared:
            problems.add("floor_platforms_mismatch:swift")

    surfaces = {surface["id"]: surface for surface in matrix["surfaces"]}
    for entry in iter_entries(matrix):
        platform = entry["platform"]
        name = platform["name"]
        lineage = [entry["surface"], *surfaces[entry["surface"]].get("builds_on", [])]
        relevant = [
            floor
            for floor in floors
            if floor["bound"] == "min"
            and floor["platform"] == name
            and floor["surface"] in lineage
        ]
        if not relevant or name == ANY:
            continue
        minimum = max(version_key(floor["version"]) for floor in relevant)
        low = platform.get("min")
        high = platform.get("max")
        if entry.get("reason") == "below-platform-floor":
            if low is not None or high is None or version_key(high) >= minimum:
                problems.add(f"entry_floor_mismatch:{entry['id']}")
            continue
        if low is None or version_key(low) < minimum:
            problems.add(f"entry_below_floor:{entry['id']}")


def _check_mlx_linkage(
    problems: set[str], matrix: Mapping[str, Any], root: Path
) -> None:
    try:
        linked = swift_mlx_platforms(_read(root, "Package.swift"))
    except ManifestError:
        problems.add("mlx_linkage_unreadable")
        return
    for entry in iter_entries(matrix):
        if entry["runtime"] != "mlx-swift":
            continue
        name = entry["platform"]["name"]
        if entry["status"] != UNSUPPORTED and name not in linked:
            problems.add(f"cell_runtime_not_linked:{entry['id']}")
        if entry.get("reason") == "runtime-not-linked" and name in linked:
            problems.add(f"unsupported_runtime_is_linked:{entry['id']}")


def _check_pins(problems: set[str], matrix: Mapping[str, Any], root: Path) -> None:
    pins = matrix["runtime_pins"]
    for pin in pins:
        for source in pin["sources"]:
            try:
                actual = pin_value(root, source, pin)
            except ManifestError:
                problems.add(f"pin_unreadable:{pin['id']}")
                continue
            if actual != pin["version"].replace(" ", ""):
                problems.add(f"pin_mismatch:{pin['id']}")
    ort_sources = {
        source
        for pin in pins
        if pin["runtime"] == "onnxruntime"
        for source in pin["sources"]
    }
    for relative in manifest_paths(root):
        if relative in ort_sources:
            continue
        try:
            text = _read(root, relative)
        except ManifestError:
            continue
        if _ORT_PIN_LITERAL.search(text):
            problems.add(f"runtime_pin_unlisted:onnxruntime:{relative}")


def _check_skew(problems: set[str], matrix: Mapping[str, Any], root: Path) -> None:
    by_runtime: dict[str, list[Mapping[str, Any]]] = {}
    for pin in matrix["runtime_pins"]:
        if pin["pinning"] != "system":
            by_runtime.setdefault(pin["runtime"], []).append(pin)
    declared = {skew["runtime"]: skew for skew in matrix["runtime_skew"]}
    for runtime, members in sorted(by_runtime.items()):
        versions = {member["version"] for member in members}
        skew = declared.get(runtime)
        if len(versions) > 1:
            if skew is None:
                problems.add(f"runtime_skew_missing:{runtime}")
            elif set(skew["pins"]) != {member["id"] for member in members}:
                problems.add(f"runtime_skew_incomplete:{runtime}")
        elif skew is not None:
            problems.add(f"runtime_skew_stale:{runtime}")
    for runtime in declared:
        if runtime not in by_runtime:
            problems.add(f"runtime_skew_stale:{runtime}")


def _check_docs(
    problems: set[str],
    matrix: Mapping[str, Any],
    root: Path,
    tiers: Mapping[str, Mapping[str, Any]],
) -> None:
    try:
        text = (root / DOC_PATH).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        problems.add("docs_missing")
        return
    expected = render_kit_support_matrix_markdown(matrix, tiers=tiers)
    try:
        current = extract_docs_block(text)
    except ManifestError:
        problems.add("docs_markers_missing")
        return
    if current != expected:
        problems.add("docs_out_of_sync")
    try:
        nav = (root / MKDOCS).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        problems.add("docs_nav_unreadable")
        return
    if DOC_PATH.relative_to("docs").as_posix() not in nav:
        problems.add("docs_not_in_nav")


def extract_docs_block(text: str) -> str:
    """Return the generated block between the matrix markers.

    Args:
        text: Docs page source.

    Returns:
        Text strictly between the begin and end markers, without the newline
        that follows the begin marker.

    Raises:
        ManifestError: If either marker is missing or out of order.
    """
    begin = text.find(DOC_BEGIN_MARKER)
    end = text.find(DOC_END_MARKER)
    if begin < 0 or end < begin:
        raise ManifestError("docs_markers_missing")
    return text[begin + len(DOC_BEGIN_MARKER) : end].removeprefix("\n")


def replace_docs_block(text: str, block: str) -> str:
    """Return the page with its generated block replaced.

    Args:
        text: Docs page source containing both markers.
        block: Rendered matrix Markdown.

    Returns:
        The updated page source.

    Raises:
        ManifestError: If either marker is missing or out of order.
    """
    extract_docs_block(text)
    begin = text.find(DOC_BEGIN_MARKER) + len(DOC_BEGIN_MARKER)
    end = text.find(DOC_END_MARKER)
    return text[:begin] + "\n" + block + text[end:]


def _summary(matrix: Mapping[str, Any]) -> str:
    counts: dict[str, int] = {}
    for entry in iter_entries(matrix):
        counts[entry["status"]] = counts.get(entry["status"], 0) + 1
    return ", ".join(f"{counts.get(key, 0)} {key}" for key in sorted(counts))


def main(argv: Sequence[str] | None = None) -> int:
    """Run the coherence check and optionally regenerate the docs block.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit status: ``0`` when coherent, ``1`` otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--write-docs",
        action="store_true",
        help="Regenerate the matrix block in docs/openmedkit-support-matrix.md.",
    )
    args = parser.parse_args(argv)
    try:
        matrix = load_kit_support_matrix(DEFAULT_KIT_SUPPORT_MATRIX, validate=False)
    except ValueError as exc:
        print(f"kit support matrix: {exc}")
        return 1
    if args.write_docs and not validate_kit_support_matrix(matrix):
        page = ROOT / DOC_PATH
        try:
            updated = replace_docs_block(
                page.read_text(encoding="utf-8"),
                render_kit_support_matrix_markdown(matrix),
            )
        except (OSError, ManifestError):
            print("kit support matrix: docs_markers_missing")
            return 1
        page.write_text(updated, encoding="utf-8")
    problems = check_repository_coherence(matrix, ROOT)
    for problem in problems:
        print(f"kit support matrix: {problem}")
    if problems:
        return 1
    print(f"kit support matrix: ok ({_summary(matrix)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
