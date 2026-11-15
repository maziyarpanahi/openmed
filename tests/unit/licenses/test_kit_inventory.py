"""Tests for the offline OpenMedKit dependency license gate."""

from __future__ import annotations

import importlib.util
import json
import shutil
import socket
import subprocess
import sys
from pathlib import Path, PurePosixPath

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "licenses" / "kit_inventory.py"
SPEC = importlib.util.spec_from_file_location("kit_license_inventory", SCRIPT)
assert SPEC is not None
assert SPEC.loader is not None
kit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = kit
SPEC.loader.exec_module(kit)

SYNTHETIC_MANIFESTS = (
    "Package.swift",
    "Package.resolved",
    "android/settings.gradle.kts",
    "js/kit-web/package-lock.json",
    "js/kit-tauri/src-tauri/Cargo.lock",
    "js/kit-flutter/pubspec.yaml",
    "js/kit-flutter/pubspec.lock",
    "js/kit-flutter/ios/kit.podspec",
)

MLKIT = "com.google.mlkit:text-recognition"
MLKIT_TERMS = "LicenseRef-Proprietary-Google-ML-Kit-Terms"

REVIEW_ROWS = {
    ("swiftpm", "swift-runtime"): "MIT",
    ("swiftpm", "swift-test-helpers"): "Apache-2.0",
    ("swiftpm", "swift-transitive"): "Apache-2.0 WITH Swift-exception",
    ("gradle", "com.example:runtime-lib"): "Apache-2.0",
    ("gradle", "com.example:bundled-lib"): "MIT",
    ("gradle", "com.example:bom"): "Apache-2.0",
    ("gradle", "com.example:literal-runtime"): "BSD-3-Clause",
    ("gradle", "com.example:test-lib"): "MIT",
    ("gradle", "com.example:pinned-only"): "MIT",
    ("gradle", "com.example:debug-tool"): "Apache-2.0",
    ("gradle", "com.example.tools:gradle-plugin"): "Apache-2.0",
    ("gradle", "junit:junit"): "EPL-1.0",
    ("gradle", MLKIT): MLKIT_TERMS,
    ("gradle-plugin", "com.android.library"): "Apache-2.0",
    ("gradle-plugin", "com.android.application"): "Apache-2.0",
    ("npm", "prod-runtime"): "MIT",
    ("npm", "nested-runtime"): "ISC",
    ("npm", "peer-runtime"): "Apache-2.0 OR MIT",
    ("npm", "dev-tool"): "MIT",
    ("cargo", "runtime-crate"): "MIT OR Apache-2.0",
    ("cargo", "shared-crate"): "Unicode-3.0",
    ("cargo", "dev-crate"): "MIT",
    ("cargo", "build-crate"): "MIT",
    ("pub", "pub_runtime"): "BSD-3-Clause",
    ("pub", "git_runtime"): "BSD-3-Clause",
    ("pub", "flutter"): "BSD-3-Clause",
    ("pub", "pub_transitive"): "BSD-3-Clause",
    ("pub", "pub_dev_tool"): "BSD-3-Clause",
    ("cocoapods", "podruntime"): "MIT",
    ("cocoapods", "podtesthelper"): "MIT",
}

SWIFT_MANIFEST = """\
// swift-tools-version: 5.10
import PackageDescription

// .package(url: "https://example.invalid/acme/commented-out.git", exact: "1.0.0")
let package = Package(
    name: "SyntheticKit",
    dependencies: [
        .package(url: "https://example.invalid/acme/swift-runtime.git", exact: "1.0.0"),
        .package(
            url: "https://example.invalid/acme/swift-test-helpers.git",
            from: "2.0.0"
        ),
        .package(path: "../LocalKit"),
    ],
    targets: [
        .target(
            name: "SyntheticKit",
            dependencies: [.product(name: "Runtime", package: "swift-runtime")]
        ),
        .testTarget(
            name: "SyntheticKitTests",
            dependencies: [
                "SyntheticKit",
                .product(name: "Helpers", package: "swift-test-helpers"),
            ]
        ),
    ]
)
"""

SETTINGS = """\
pluginManagement {
    resolutionStrategy {
        eachPlugin {
            useModule("com.example.tools:gradle-plugin:${requested.version}")
        }
    }
}
rootProject.name = "synthetic"
include(":core", ":openmedkit-mlkit")
include(":demo:app")
// include(":disabled")
"""

CATALOG = """\
[versions]
runtime = "1.0.0"

[libraries]
runtime-lib = { module = "com.example:runtime-lib", version.ref = "runtime" }
bundled-lib = { group = "com.example", name = "bundled-lib", version = "1.0.0" }
test-lib = "com.example:test-lib:1.0.0"
pinned-only = { module = "com.example:pinned-only", version = "1.0.0" }
mlkit-text = { module = "com.google.mlkit:text-recognition", version = "16.0.1" }

[bundles]
runtime-bundle = ["bundled-lib"]

[plugins]
android-library = { id = "com.android.library", version = "8.5.1" }
"""

CORE_BUILD = """\
plugins {
    alias(libs.plugins.android.library)
    `maven-publish`
}

dependencies {
    implementation(libs.runtime.lib)
    implementation(libs.bundles.runtime.bundle)
    implementation(platform("com.example:bom:1.0.0"))
    api("com.example:literal-runtime:2.0.0")
    testImplementation(libs.test.lib)
    testImplementation("junit:junit:4.13.2")
    // implementation("com.example:commented-out:1.0.0")
}
"""

MLKIT_BUILD = """\
plugins {
    id("com.android.library")
}

dependencies {
    api(project(":core"))
    api(libs.mlkit.text)
}
"""

DEMO_BUILD = """\
plugins {
    id("com.android.application")
}

dependencies {
    implementation(project(":core"))
    debugImplementation("com.example:debug-tool:1.0.0")
}
"""

NPM_PACKAGE = {
    "name": "kit-web",
    "dependencies": {"prod-runtime": "^1.0.0", "local-kit": "file:../local"},
    "peerDependencies": {"peer-runtime": ">=1.0.0"},
    "devDependencies": {"dev-tool": "^1.0.0"},
}

NPM_LOCK = {
    "name": "kit-web",
    "lockfileVersion": 3,
    "packages": {
        "": NPM_PACKAGE,
        "../local": {"version": "0.0.0"},
        "node_modules/local-kit": {"resolved": "../local", "link": True},
        "node_modules/prod-runtime": {"version": "1.0.0", "license": "MIT"},
        "node_modules/prod-runtime/node_modules/nested-runtime": {"version": "1.0.0"},
        "node_modules/dev-tool": {"version": "1.0.0", "dev": True},
        "node_modules/dev-transitive": {"version": "1.0.0", "dev": True},
        "node_modules/@scope/dev-transitive": {"version": "1.0.0", "dev": True},
    },
}

CARGO_TOML = """\
[package]
name = "synthetic-sidecar"
version = "0.1.0"

[dependencies]
runtime-crate = "1"

[dev-dependencies]
dev-crate = "1"

[build-dependencies]
build-crate = "1"
"""

CARGO_LOCK = """\
version = 4

[[package]]
name = "synthetic-sidecar"
version = "0.1.0"
dependencies = ["build-crate", "dev-crate", "runtime-crate"]

[[package]]
name = "runtime-crate"
version = "1.0.0"
source = "registry+https://example.invalid/index"
dependencies = ["shared-crate"]

[[package]]
name = "shared-crate"
version = "1.0.0"
source = "registry+https://example.invalid/index"

[[package]]
name = "dev-crate"
version = "1.0.0"
source = "registry+https://example.invalid/index"
dependencies = ["dev-only-transitive", "shared-crate"]

[[package]]
name = "dev-only-transitive"
version = "1.0.0"
source = "registry+https://example.invalid/index"

[[package]]
name = "build-crate"
version = "1.0.0"
source = "registry+https://example.invalid/index"
"""

PUBSPEC = """\
name: kit_flutter
description: Synthetic plugin.
publish_to: none

dependencies:
  pub_runtime: ^1.0.0
  flutter:
    sdk: flutter
  local_plugin:
    path: ../local
  git_runtime:
    git:
      url: https://example.invalid/git_runtime.git
      path: packages/git_runtime

dev_dependencies:
  pub_dev_tool: ^1.0.0

flutter:
  plugin:
    platforms:
      ios:
        pluginClass: KitPlugin
"""

PUBSPEC_LOCK = """\
# Generated by pub
packages:
  pub_runtime:
    dependency: "direct main"
    description:
      name: pub_runtime
      url: "https://example.invalid"
    source: hosted
    version: "1.0.0"
  pub_transitive:
    dependency: transitive
    source: hosted
    version: "1.0.0"
  pub_dev_tool:
    dependency: "direct dev"
    source: hosted
    version: "1.0.0"
  local_plugin:
    dependency: "direct main"
    description:
      path: "../local"
    source: path
    version: "0.0.1"
sdks:
  dart: ">=3.0.0 <4.0.0"
"""

PODSPEC = """\
Pod::Spec.new do |s|
  s.name = 'Kit'
  # s.dependency 'CommentedOut'
  s.dependency 'PodRuntime', '1.0.0'
  s.test_spec 'Tests' do |test_spec|
    test_spec.dependency 'PodTestHelper'
  end
end
"""


def swift_resolved(pins: list[dict[str, object]]) -> str:
    return json.dumps({"pins": pins, "version": 3}, indent=2)


DEFAULT_PINS = [
    {
        "identity": "swift-runtime",
        "kind": "remoteSourceControl",
        "location": "https://example.invalid/acme/swift-runtime.git",
        "state": {"version": "1.0.0"},
    },
    {
        "identity": "swift-test-helpers",
        "kind": "remoteSourceControl",
        "location": "https://example.invalid/acme/swift-test-helpers.git",
        "state": {"version": "2.0.0"},
    },
    {
        "identity": "swift-transitive",
        "kind": "remoteSourceControl",
        "location": "https://example.invalid/acme/swift-transitive",
        "state": {"version": "1.0.0"},
    },
    {"identity": "localkit", "kind": "fileSystem", "location": "../LocalKit"},
]


def write(root: Path, relative: str, content: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def review_table(rows: dict[tuple[str, str], str]) -> str:
    lines = [
        "# Synthetic kit inventory",
        "",
        "| Ecosystem | Package | License expression | Review note |",
        "| --- | --- | --- | --- |",
    ]
    for (ecosystem, name), expression in sorted(rows.items()):
        lines.append(f"| {ecosystem} | `{name}` | `{expression}` | |")
    return "\n".join(lines) + "\n"


def build_repo(root: Path, rows: dict[tuple[str, str], str] | None = None) -> Path:
    write(root, "Package.swift", SWIFT_MANIFEST)
    write(root, "Package.resolved", swift_resolved(DEFAULT_PINS))
    write(root, "android/settings.gradle.kts", SETTINGS)
    write(root, "android/gradle/libs.versions.toml", CATALOG)
    write(
        root,
        "android/build.gradle.kts",
        "plugins {\n    alias(libs.plugins.android.library) apply false\n}\n",
    )
    write(root, "android/core/build.gradle.kts", CORE_BUILD)
    write(root, "android/openmedkit-mlkit/build.gradle.kts", MLKIT_BUILD)
    write(root, "android/demo/app/build.gradle.kts", DEMO_BUILD)
    write(root, "js/kit-web/package.json", json.dumps(NPM_PACKAGE, indent=2))
    write(root, "js/kit-web/package-lock.json", json.dumps(NPM_LOCK, indent=2))
    write(root, "js/kit-tauri/src-tauri/Cargo.toml", CARGO_TOML)
    write(root, "js/kit-tauri/src-tauri/Cargo.lock", CARGO_LOCK)
    write(root, "js/kit-flutter/pubspec.yaml", PUBSPEC)
    write(root, "js/kit-flutter/pubspec.lock", PUBSPEC_LOCK)
    write(root, "js/kit-flutter/ios/kit.podspec", PODSPEC)
    return write(
        root,
        "docs/kit-license-inventory.md",
        review_table(REVIEW_ROWS if rows is None else rows),
    )


def run_gate(root: Path, inventory_path: Path) -> tuple[int, Path]:
    report = root / "out" / "report.json"
    argv = ["--root", str(root), "--inventory", str(inventory_path)]
    for manifest in SYNTHETIC_MANIFESTS:
        argv.extend(["--manifest", manifest])
    argv.extend(["--report", str(report)])
    return kit.main(argv), report


def records_by_key(audit) -> dict[tuple[str, str, str], object]:
    return {
        (
            record.dependency.ecosystem,
            record.dependency.name,
            record.dependency.manifest,
        ): record
        for record in audit.records
    }


def test_checked_in_kits_pass_and_report_mlkit_as_restricted_exception() -> None:
    audit = kit.audit_kits()

    assert not audit.failures
    exceptions = {
        (record.dependency.name, record.dependency.manifest): record
        for record in audit.exceptions
    }
    assert exceptions
    assert {name for name, _manifest in exceptions} == {MLKIT}
    # Holds both before and after #4451 moves ML Kit to :openmedkit-mlkit.
    assert {manifest for _name, manifest in exceptions} <= (
        kit.SCOPED_KIT_EXCEPTIONS[0].manifests
    )
    assert all(
        record.classification == kit.LicenseClass.RESTRICTED
        and record.exception_id == "mlkit-optional-ocr"
        for record in exceptions.values()
    )

    records = records_by_key(audit)
    for manifest in ("Package.swift", "swift/OpenMedKit/Package.swift"):
        whisperkit = records[("swiftpm", "argmax-oss-swift", manifest)]
        assert whisperkit.decision == kit.Decision.PERMISSIVE
        assert whisperkit.dependency.usage == kit.Usage.RUNTIME
    assert (
        records[
            ("gradle", "junit:junit", "android/openmedkit/build.gradle.kts")
        ].dependency.usage
        == kit.Usage.DEVELOPMENT
    )
    assert (
        records[
            ("npm", "typescript", "js/openmedkit-web/package-lock.json")
        ].dependency.usage
        == kit.Usage.DEVELOPMENT
    )
    assert (
        records[
            ("npm", "onnxruntime-web", "js/openmedkit-web/package-lock.json")
        ].dependency.usage
        == kit.Usage.RUNTIME
    )
    assert audit.counted_development.get("npm", 0) > 0
    assert audit.unused_review_rows == 0
    assert {record.dependency.ecosystem for record in audit.records} == kit.ECOSYSTEMS


def test_every_committed_kit_manifest_is_read_by_the_gate() -> None:
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("git checkout metadata is unavailable")
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip("git ls-files is unavailable")

    names = {
        "Package.swift",
        "Package.resolved",
        "package-lock.json",
        "package.json",
        "Cargo.lock",
        "Cargo.toml",
        "pubspec.yaml",
        "pubspec.lock",
        "libs.versions.toml",
        "build.gradle",
        "build.gradle.kts",
        "settings.gradle",
        "settings.gradle.kts",
        "Podfile",
        "Podfile.lock",
    }
    committed = set()
    for raw in result.stdout.decode("utf-8").split("\0"):
        if not raw:
            continue
        path = PurePosixPath(raw)
        is_manifest = path.name in names or path.suffix == ".podspec"
        in_kit = len(path.parts) == 1 or path.parts[0] in {"android", "js", "swift"}
        if is_manifest and in_kit and "node_modules" not in path.parts:
            committed.add(raw)

    discovery = kit.discover_kit_dependencies()
    assert committed
    assert committed <= discovery.manifests_read, sorted(
        committed - discovery.manifests_read
    )


def test_synthetic_kits_distinguish_runtime_and_development(tmp_path: Path) -> None:
    inventory_path = build_repo(tmp_path)

    audit = kit.audit_kits(tmp_path, inventory_path, SYNTHETIC_MANIFESTS)

    assert not audit.failures
    usage = {
        (record.dependency.ecosystem, record.dependency.name): record.dependency.usage
        for record in audit.records
    }
    runtime = kit.Usage.RUNTIME
    development = kit.Usage.DEVELOPMENT
    assert usage == {
        ("cargo", "build-crate"): development,
        ("cargo", "dev-crate"): development,
        ("cargo", "runtime-crate"): runtime,
        ("cargo", "shared-crate"): runtime,
        ("cocoapods", "podruntime"): runtime,
        ("cocoapods", "podtesthelper"): development,
        ("gradle", "com.example.tools:gradle-plugin"): development,
        ("gradle", "com.example:bom"): runtime,
        ("gradle", "com.example:bundled-lib"): runtime,
        ("gradle", "com.example:debug-tool"): development,
        ("gradle", "com.example:literal-runtime"): runtime,
        ("gradle", "com.example:pinned-only"): runtime,
        ("gradle", "com.example:runtime-lib"): runtime,
        ("gradle", "com.example:test-lib"): development,
        ("gradle", "junit:junit"): development,
        ("gradle", MLKIT): runtime,
        ("gradle-plugin", "com.android.application"): development,
        ("gradle-plugin", "com.android.library"): development,
        ("npm", "dev-tool"): development,
        ("npm", "nested-runtime"): runtime,
        ("npm", "peer-runtime"): runtime,
        ("npm", "prod-runtime"): runtime,
        ("pub", "flutter"): runtime,
        ("pub", "git_runtime"): runtime,
        ("pub", "pub_dev_tool"): development,
        ("pub", "pub_runtime"): runtime,
        ("pub", "pub_transitive"): runtime,
        ("swiftpm", "swift-runtime"): runtime,
        ("swiftpm", "swift-test-helpers"): development,
        ("swiftpm", "swift-transitive"): runtime,
    }
    assert audit.counted_development == {"cargo": 1, "npm": 2}
    records = records_by_key(audit)
    assert not records[
        ("npm", "nested-runtime", "js/kit-web/package-lock.json")
    ].dependency.direct
    assert records[
        ("npm", "peer-runtime", "js/kit-web/package-lock.json")
    ].dependency.direct
    assert (
        records[("gradle", "junit:junit", "android/core/build.gradle.kts")].decision
        == kit.Decision.DEVELOPMENT_ONLY
    )
    mlkit = records[("gradle", MLKIT, "android/openmedkit-mlkit/build.gradle.kts")]
    assert mlkit.decision == kit.Decision.SCOPED_EXCEPTION
    assert mlkit.classification == kit.LicenseClass.RESTRICTED


@pytest.mark.parametrize(
    ("manifest", "old", "new", "ecosystem", "package", "usage"),
    [
        (
            "Package.swift",
            "        .package(path: ",
            '        .package(url: "https://example.invalid/acme/unreviewed-swift.git",'
            ' exact: "1.0.0"),\n        .package(path: ',
            "swiftpm",
            "unreviewed-swift",
            "runtime",
        ),
        (
            "Package.resolved",
            '"pins": [',
            '"pins": [{"identity": "unreviewed-pin", "kind": "remoteSourceControl",'
            ' "location": "https://example.invalid/unreviewed-pin.git"},',
            "swiftpm",
            "unreviewed-pin",
            "runtime",
        ),
        (
            "android/gradle/libs.versions.toml",
            "[bundles]",
            'unreviewed-lib = "com.example:unreviewed-lib:1.0.0"\n\n[bundles]',
            "gradle",
            "com.example:unreviewed-lib",
            "runtime",
        ),
        (
            "android/core/build.gradle.kts",
            "    api(",
            '    runtimeOnly("com.example:unreviewed-literal:1.0.0")\n    api(',
            "gradle",
            "com.example:unreviewed-literal",
            "runtime",
        ),
        (
            "android/demo/app/build.gradle.kts",
            "    debugImplementation(",
            '    testImplementation("com.example:unreviewed-test:1.0.0")\n'
            "    debugImplementation(",
            "gradle",
            "com.example:unreviewed-test",
            "development",
        ),
        (
            "js/kit-web/package-lock.json",
            '"node_modules/dev-tool": {',
            '"node_modules/unreviewed-transitive": {"version": "1.0.0"},\n'
            '    "node_modules/dev-tool": {',
            "npm",
            "unreviewed-transitive",
            "runtime",
        ),
        (
            "js/kit-web/package.json",
            '"peerDependencies": {',
            '"peerDependencies": {\n    "unreviewed-peer": ">=1",',
            "npm",
            "unreviewed-peer",
            "runtime",
        ),
        (
            "js/kit-tauri/src-tauri/Cargo.lock",
            'name = "build-crate"\n',
            'name = "unreviewed-crate"\nversion = "0.1.0"\n'
            'source = "registry+https://example.invalid/index"\n\n[[package]]\n'
            'name = "build-crate"\n',
            "cargo",
            "unreviewed-crate",
            "runtime",
        ),
        (
            "js/kit-flutter/pubspec.yaml",
            "dev_dependencies:\n",
            "dev_dependencies:\n  unreviewed_dev: ^1.0.0\n",
            "pub",
            "unreviewed_dev",
            "development",
        ),
        (
            "js/kit-flutter/pubspec.lock",
            "sdks:",
            "  unreviewed_locked:\n    dependency: transitive\n    source: hosted\n"
            "sdks:",
            "pub",
            "unreviewed_locked",
            "runtime",
        ),
        (
            "js/kit-flutter/ios/kit.podspec",
            "  s.test_spec",
            "  s.dependency 'UnreviewedPod'\n  s.test_spec",
            "cocoapods",
            "unreviewedpod",
            "runtime",
        ),
    ],
)
def test_unreviewed_dependency_fails_naming_only_manifest_and_package(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    manifest: str,
    old: str,
    new: str,
    ecosystem: str,
    package: str,
    usage: str,
) -> None:
    inventory_path = build_repo(tmp_path)
    target = tmp_path / manifest
    original = target.read_text(encoding="utf-8")
    assert old in original
    target.write_text(original.replace(old, new, 1), encoding="utf-8")

    status, report_path = run_gate(tmp_path, inventory_path)
    output = capsys.readouterr()

    assert status == 1
    owner = (
        "js/kit-web/package.json" if manifest == "js/kit-web/package.json" else manifest
    )
    expected = f"- {owner}: {ecosystem} {package} ({usage}): unreviewed"
    failures = [line for line in output.err.splitlines() if line.startswith("- ")]
    assert failures == [expected]
    assert str(tmp_path) not in output.out + output.err
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["summary"]["failures"] == 1
    assert str(tmp_path) not in report_path.read_text(encoding="utf-8")


def test_mlkit_exception_is_limited_to_the_optional_module_paths(
    tmp_path: Path,
) -> None:
    case_root = tmp_path
    cases = {
        "core-module": (
            "android/core/build.gradle.kts",
            "    api(",
            "    implementation(libs.mlkit.text)\n    api(",
        ),
        "flutter-android": (
            "js/kit-flutter/android/build.gradle",
            None,
            "dependencies {\n"
            "    implementation 'com.google.mlkit:text-recognition:16.0.1'\n}\n",
        ),
    }
    for label, (manifest, old, new) in cases.items():
        root = case_root / label
        inventory_path = build_repo(root)
        target = root / manifest
        if old is None:
            write(root, manifest, new)
        else:
            text = target.read_text(encoding="utf-8")
            target.write_text(text.replace(old, new, 1), encoding="utf-8")
        manifests = SYNTHETIC_MANIFESTS + (
            (manifest,) if manifest.startswith("js/kit-flutter/android") else ()
        )

        audit = kit.audit_kits(root, inventory_path, manifests)

        assert [
            (record.dependency.name, record.dependency.manifest, record.decision)
            for record in audit.failures
        ] == [(MLKIT, manifest, kit.Decision.DENIED)], label
        assert [record.dependency.manifest for record in audit.exceptions] == [
            "android/openmedkit-mlkit/build.gradle.kts"
        ]


def test_mlkit_exception_requires_the_reviewed_restricted_expression(
    tmp_path: Path,
) -> None:
    case_root = tmp_path
    for label, expression in {
        "unknown": "LicenseRef-Google-ML-Kit",
        "other-restricted": "LicenseRef-Proprietary-Other",
        "missing": None,
    }.items():
        rows = dict(REVIEW_ROWS)
        if expression is None:
            del rows[("gradle", MLKIT)]
        else:
            rows[("gradle", MLKIT)] = expression
        root = case_root / label
        inventory_path = build_repo(root, rows)

        audit = kit.audit_kits(root, inventory_path, SYNTHETIC_MANIFESTS)

        assert [record.dependency.name for record in audit.failures] == [MLKIT]
        assert not audit.exceptions


def test_exception_use_is_printed_and_never_silent(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    inventory_path = build_repo(tmp_path)

    status, report_path = run_gate(tmp_path, inventory_path)
    output = capsys.readouterr()

    assert status == 0
    assert (
        "scoped exception mlkit-optional-ocr: android/openmedkit-mlkit/"
        "build.gradle.kts: gradle com.google.mlkit:text-recognition (runtime) is "
        "restricted"
    ) in output.out
    report = json.loads(report_path.read_text(encoding="utf-8"))
    excepted = [
        entry
        for entry in report["entries"]
        if entry.get("exception") == "mlkit-optional-ocr"
    ]
    assert excepted == [
        {
            "classification": "restricted",
            "decision": "scoped-exception",
            "direct": True,
            "ecosystem": "gradle",
            "exception": "mlkit-optional-ocr",
            "manifest": "android/openmedkit-mlkit/build.gradle.kts",
            "package": MLKIT,
            "usage": "runtime",
        }
    ]
    assert report["summary"]["decisions"]["scoped-exception"] == 1


def test_development_only_licenses_never_cover_runtime_or_strong_copyleft(
    tmp_path: Path,
) -> None:
    case_root = tmp_path
    cases = {
        "epl-at-runtime": (
            "android/core/build.gradle.kts",
            'testImplementation("junit:junit:4.13.2")',
            'implementation("junit:junit:4.13.2")',
            REVIEW_ROWS,
            ("gradle", "junit:junit"),
        ),
        "gpl-for-development": (
            "android/core/build.gradle.kts",
            "testImplementation(libs.test.lib)",
            "testImplementation(libs.test.lib)",
            {**REVIEW_ROWS, ("gradle", "com.example:test-lib"): "GPL-3.0-only"},
            ("gradle", "com.example:test-lib"),
        ),
        "unknown-for-development": (
            "js/kit-web/package-lock.json",
            '"node_modules/dev-tool"',
            '"node_modules/dev-tool"',
            {**REVIEW_ROWS, ("npm", "dev-tool"): "LicenseRef-unreviewed"},
            ("npm", "dev-tool"),
        ),
    }
    for label, (manifest, old, new, rows, failing) in cases.items():
        root = case_root / label
        inventory_path = build_repo(root, rows)
        target = root / manifest
        target.write_text(
            target.read_text(encoding="utf-8").replace(old, new, 1),
            encoding="utf-8",
        )

        audit = kit.audit_kits(root, inventory_path, SYNTHETIC_MANIFESTS)

        assert [
            (record.dependency.ecosystem, record.dependency.name, record.decision)
            for record in audit.failures
        ] == [(*failing, kit.Decision.DENIED)], label


def test_report_and_cli_exclude_license_text_versions_and_local_paths(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    private_marker = "LicenseRef-private-example-001"
    rows = {**REVIEW_ROWS, ("npm", "prod-runtime"): private_marker}
    inventory_path = build_repo(tmp_path, rows)

    status, report_path = run_gate(tmp_path, inventory_path)
    output = capsys.readouterr()
    report_text = report_path.read_text(encoding="utf-8")

    assert status == 1
    for text in (output.out, output.err, report_text):
        assert private_marker not in text
        assert str(tmp_path) not in text
        assert "example.invalid" not in text
        assert "16.0.1" not in text
    report = json.loads(report_text)
    assert report["schema_version"] == 1
    assert all(
        not PurePosixPath(entry["manifest"]).is_absolute()
        for entry in report["entries"]
    )
    assert {key for entry in report["entries"] for key in entry} <= {
        "classification",
        "decision",
        "direct",
        "ecosystem",
        "exception",
        "manifest",
        "package",
        "usage",
    }


def test_gate_makes_no_network_calls(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def refuse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("the kit license gate attempted network access")

    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)

    assert kit.main([]) == 0
    assert "kit license gate passed" in capsys.readouterr().out


def test_missing_registered_manifest_fails_with_relative_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    inventory_path = build_repo(tmp_path)
    (tmp_path / "js/kit-flutter/ios/kit.podspec").unlink()

    status, _report = run_gate(tmp_path, inventory_path)
    error = capsys.readouterr().err

    assert status == 1
    assert "registered kit manifest is missing: js/kit-flutter/ios/kit.podspec" in error
    assert str(tmp_path) not in error


def test_unresolvable_gradle_declarations_fail_closed(tmp_path: Path) -> None:
    case_root = tmp_path
    for label, declaration, reason in [
        (
            "local-binary",
            'implementation(files("libs/vendor.aar"))',
            "local binary dependency",
        ),
        (
            "variable-coordinate",
            'implementation("$vendorGroup:vendor:1.0.0")',
            "not statically resolvable",
        ),
        ("unknown-alias", "implementation(libs.not.declared)", "alias is undeclared"),
    ]:
        root = case_root / label
        inventory_path = build_repo(root)
        target = root / "android/core/build.gradle.kts"
        target.write_text(
            target.read_text(encoding="utf-8").replace(
                "    api(", f"    {declaration}\n    api(", 1
            ),
            encoding="utf-8",
        )

        with pytest.raises(kit.KitManifestError, match=reason) as error:
            kit.audit_kits(root, inventory_path, SYNTHETIC_MANIFESTS)

        assert error.value.manifest == "android/core/build.gradle.kts"
        assert str(root) not in str(error.value)


def test_review_table_rejects_duplicate_and_unknown_rows(tmp_path: Path) -> None:
    header = "| Ecosystem | Package | License expression |\n| --- | --- | --- |\n"
    for label, rows, reason in [
        ("duplicate", "| npm | `a` | `MIT` |\n| npm | `a` | `MIT` |\n", "duplicate"),
        ("ecosystem", "| maven | `a` | `MIT` |\n", "unknown ecosystem"),
        ("name", "| npm | `a b` | `MIT` |\n", "invalid package name"),
        ("empty", "No table here.\n", "no review table"),
    ]:
        path = tmp_path / f"{label}.md"
        path.write_text(header + rows if label != "empty" else rows, encoding="utf-8")
        with pytest.raises(kit.InventoryError, match=reason):
            kit.parse_kit_inventory(path)


def test_legacy_lockfile_formats_are_supported(tmp_path: Path) -> None:
    write(
        tmp_path,
        "Package.resolved",
        json.dumps(
            {
                "object": {
                    "pins": [
                        {
                            "package": "Legacy",
                            "repositoryURL": "https://example.invalid/acme/Legacy.git",
                            "state": {"version": "1.0.0"},
                        }
                    ]
                },
                "version": 1,
            }
        ),
    )
    write(
        tmp_path,
        "js/legacy/package.json",
        json.dumps({"dependencies": {"left": "1"}, "devDependencies": {"right": "1"}}),
    )
    write(
        tmp_path,
        "js/legacy/package-lock.json",
        json.dumps(
            {
                "lockfileVersion": 1,
                "dependencies": {
                    "left": {
                        "version": "1.0.0",
                        "dependencies": {"left-child": {"version": "1.0.0"}},
                    },
                    "right": {"version": "1.0.0", "dev": True},
                    "right-child": {"version": "1.0.0", "dev": True},
                },
            }
        ),
    )

    discovery = kit.discover_kit_dependencies(
        tmp_path, ("Package.resolved", "js/legacy/package-lock.json")
    )

    assert sorted(
        (dependency.ecosystem, dependency.name, dependency.usage, dependency.direct)
        for dependency in discovery.dependencies
    ) == [
        ("npm", "left", "runtime", True),
        ("npm", "left-child", "runtime", False),
        ("npm", "right", "development", True),
        ("swiftpm", "legacy", "runtime", False),
    ]
    assert discovery.counted_development == {"npm": 1}


def test_manifest_paths_must_be_repository_relative(tmp_path: Path) -> None:
    for manifest in ("/etc/Package.swift", "../Package.swift"):
        with pytest.raises(kit.InventoryError, match="repository-relative"):
            kit.discover_kit_dependencies(tmp_path, (manifest,))
