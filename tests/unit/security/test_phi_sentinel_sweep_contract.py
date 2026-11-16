"""Packaging and parity contract for the OpenMedKit PHI sentinel sweep."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).parents[3]
SWIFT_MANIFESTS = (
    ROOT / "Package.swift",
    ROOT / "swift" / "OpenMedKit" / "Package.swift",
)
SWIFT_SWEEP = ROOT / "swift" / "OpenMedKit" / "Sources" / "OpenMedPrivacySweep"
ANDROID_BUILD = ROOT / "android" / "openmedkit" / "build.gradle.kts"
KOTLIN_SWEEP = (
    ROOT
    / "android"
    / "openmedkit"
    / "src"
    / "sharedTest"
    / "kotlin"
    / "com"
    / "openmed"
    / "openmedkit"
    / "privacy"
)
DOC = ROOT / "docs" / "security" / "no-raw-phi-logging.md"

BOUND_NAMES = (
    "maxFiles",
    "maxDepth",
    "maxBytesPerFile",
    "maxTotalBytes",
    "maxDuration",
    "maxLogEntries",
    "maxLogBytes",
)


def _number(raw: str) -> float:
    return float(raw.replace("_", "").rstrip("L"))


def _swift_bounds(name: str) -> dict[str, float]:
    source = (SWIFT_SWEEP / "SweepConfiguration.swift").read_text(encoding="utf-8")
    block = source.split(f"public static let {name} = SweepBounds(", 1)[1]
    block = block.split(")", 1)[0]
    return {key: _number(value) for key, value in re.findall(r"(\w+): ([\d_]+)", block)}


def _kotlin_bounds(defaults: bool) -> dict[str, float]:
    source = (KOTLIN_SWEEP / "SweepConfiguration.kt").read_text(encoding="utf-8")
    if defaults:
        block = source.split("data class SweepBounds(", 1)[1].split(") {", 1)[0]
        pairs = re.findall(r"val (\w+): \w+ = ([\d_]+L?)", block)
    else:
        block = source.split("val CEILING = SweepBounds(", 1)[1].split(")", 1)[0]
        pairs = re.findall(r"(\w+) = ([\d_]+L?)", block)
    values = {key: _number(value) for key, value in pairs}
    values["maxDuration"] = values.pop("maxDurationMillis") / 1_000
    return values


def _enum_values(path: Path, pattern: str) -> set[str]:
    return set(re.findall(pattern, path.read_text(encoding="utf-8")))


def test_swift_sweep_is_a_test_support_target_no_product_exposes() -> None:
    for manifest in SWIFT_MANIFESTS:
        source = manifest.read_text(encoding="utf-8")
        assert '.target(\n            name: "OpenMedPrivacySweep"' in source
        products = source.split("products: [", 1)[1].split("dependencies: [", 1)[0]
        assert "OpenMedPrivacySweep" not in products
        for test_target in ("OpenMedKitTests", "ExtensionTests"):
            block = source.split(f'name: "{test_target}"', 1)[1].split(")", 1)[0]
            assert '"OpenMedPrivacySweep"' in block
        for library in ("OpenMedKit", "OpenMedExtensionSupport"):
            target = source.split(f'.target(\n            name: "{library}"', 1)[1]
            assert "OpenMedPrivacySweep" not in target.split(".target(", 1)[0]


def test_kotlin_sweep_compiles_only_into_test_source_sets() -> None:
    build = ANDROID_BUILD.read_text(encoding="utf-8")
    source_sets = build.split("sourceSets {", 1)[1].split("\n    }\n", 1)[0]
    main = source_sets.split('getByName("main") {', 1)[1].split("}", 1)[0]
    assert "sharedTest" not in main
    for name in ("test", "androidTest"):
        block = source_sets.split(f'getByName("{name}") {{', 1)[1].split("}", 1)[0]
        assert 'java.srcDir("src/sharedTest/kotlin")' in block
    assert build.count("src/sharedTest/kotlin") == 2
    main_sources = ROOT / "android" / "openmedkit" / "src" / "main"
    assert not list(main_sources.rglob("*Sentinel*"))


def test_swift_and_kotlin_bounds_match_each_other_and_the_docs() -> None:
    doc = DOC.read_text(encoding="utf-8")
    for swift_name, kotlin_defaults in (("standard", True), ("ceiling", False)):
        swift = _swift_bounds(swift_name)
        kotlin = _kotlin_bounds(kotlin_defaults)
        assert set(swift) == set(BOUND_NAMES)
        assert swift == kotlin
    standard = _swift_bounds("standard")
    for name in BOUND_NAMES:
        snake = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
        row = next(line for line in doc.splitlines() if line.startswith(f"| `{snake}`"))
        default = row.split("|")[2]
        if name == "maxDuration":
            assert f"{int(standard[name])} s" in default
        else:
            assert f"{int(standard[name]):,}" in default


def test_platform_vocabularies_match_and_are_documented() -> None:
    doc = DOC.read_text(encoding="utf-8")
    encodings = {"utf8", "utf16le", "utf16be", "base64", "base64url", "percent"}
    swift_source = (SWIFT_SWEEP / "PHISentinel.swift").read_text(encoding="utf-8")
    swift_enum = swift_source.split("public enum SentinelEncoding", 1)[1]
    swift_enum = swift_enum.split("public static func <", 1)[0]
    kotlin_source = (KOTLIN_SWEEP / "PhiSentinel.kt").read_text(encoding="utf-8")
    kotlin_enum = kotlin_source.split("enum class SentinelEncoding", 1)[1]
    kotlin_enum = kotlin_enum.split("\n}\n", 1)[0]
    assert set(re.findall(r"(?m)^    case (\w+)$", swift_enum)) == encodings
    assert set(re.findall(r'(?m)^    [A-Z0-9]+\("(\w+)"\),$', kotlin_enum)) == encodings

    kinds = {"plist", "json", "xml", "sqlite", "text", "other"}
    kotlin_kinds = _enum_values(
        KOTLIN_SWEEP / "PhiSentinelSweepReport.kt", r'[A-Z]+\("(\w+)"\)'
    )
    assert kinds <= kotlin_kinds
    for kind in kinds:
        assert f"`{kind}`" in doc

    storage = _enum_values(
        KOTLIN_SWEEP / "SweepConfiguration.kt", r'[A-Z_]+\("(\w+)"\),'
    )
    for wire_name in storage & {
        "files",
        "cache",
        "code_cache",
        "no_backup",
        "shared_prefs",
        "databases",
        "data_other",
        "external_files",
        "external_cache",
    }:
        assert f"`{wire_name}`" in doc
    swift_storage = (SWIFT_SWEEP / "SweepConfiguration.swift").read_text(
        encoding="utf-8"
    )
    for wire_name in ("application_support", "tmp", "app_group"):
        assert f'"{wire_name}"' in swift_storage
        assert f"`{wire_name}`" in doc


def test_reports_are_documented_as_value_free_and_schema_versions_match() -> None:
    swift = (SWIFT_SWEEP / "PHISentinelSweepReport.swift").read_text(encoding="utf-8")
    kotlin = (KOTLIN_SWEEP / "PhiSentinelSweepReport.kt").read_text(encoding="utf-8")
    schema = "openmed.phi_sentinel_sweep.v1"
    assert f'"{schema}"' in swift
    assert f'"{schema}"' in kotlin
    doc = DOC.read_text(encoding="utf-8")
    assert "never contain sentinel values" in doc
    assert "`clean`" in doc
    assert "`exhaustive`" in doc
