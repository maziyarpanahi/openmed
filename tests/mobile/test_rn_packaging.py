"""Packaging and native registration checks for the React Native bridge.

These checks run offline. They prove what the packed tarball contains, that
the iOS export file, the Android package and the TypeScript native interface
agree on the module name and its four promise methods, and that the podspec,
Gradle module and ``react-native.config.js`` follow the React Native CLI
autolinking contract. Checks that need ``npm``, ``node``, ``ruby`` or
CocoaPods skip when the tool is not installed.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = ROOT / "js" / "openmedkit-react-native"
PACKAGE_JSON = PACKAGE_DIR / "package.json"
PODSPEC = PACKAGE_DIR / "openmedkit-react-native.podspec"
RN_CONFIG = PACKAGE_DIR / "react-native.config.js"
IOS_DIR = PACKAGE_DIR / "ios"
EXPORT_FILE = IOS_DIR / "OpenMedKitRN.m"
SWIFT_MODULE = IOS_DIR / "OpenMedKitRN.swift"
ANDROID_DIR = PACKAGE_DIR / "android"
BUILD_GRADLE = ANDROID_DIR / "build.gradle"
KOTLIN_PACKAGE_NAME = "com.openmed.openmedkit.reactnative"
KOTLIN_DIR = ANDROID_DIR / "src/main/java/com/openmed/openmedkit/reactnative"
KOTLIN_MODULE = KOTLIN_DIR / "OpenMedKitRnModule.kt"
KOTLIN_PACKAGE = KOTLIN_DIR / "OpenMedKitRnPackage.kt"
TS_ENTRY = PACKAGE_DIR / "src" / "index.ts"

MODULE_NAME = "OpenMedKitRN"
BRIDGE_METHODS = frozenset({"loadModel", "analyzeText", "extractPii", "deidentify"})
OPENMEDKIT_RELEASE = "3.2.0"
REACT_NATIVE_BASELINE = "0.77.3"

EXPECTED_PACKED_FILES = frozenset(
    {
        "README.md",
        "android/build.gradle",
        f"android/src/main/java/{KOTLIN_PACKAGE_NAME.replace('.', '/')}/"
        "OpenMedKitRnModule.kt",
        f"android/src/main/java/{KOTLIN_PACKAGE_NAME.replace('.', '/')}/"
        "OpenMedKitRnPackage.kt",
        "ios/OpenMedKitRN.m",
        "ios/OpenMedKitRN.swift",
        "openmedkit-react-native.podspec",
        "package.json",
        "react-native.config.js",
        "src/index.ts",
    }
)

# Allowed keys under dependency.platforms.* in the React Native CLI config
# schema; the CLI rejects unknown keys inside these objects.
CLI_IOS_DEPENDENCY_KEYS = frozenset({"scriptPhases", "configurations"})
CLI_ANDROID_DEPENDENCY_KEYS = frozenset(
    {
        "sourceDir",
        "manifestPath",
        "packageName",
        "packageImportPath",
        "packageInstance",
        "dependencyConfiguration",
        "buildTypes",
        "libraryName",
        "componentDescriptors",
        "cmakeListsPath",
        "cxxModuleCMakeListsModuleName",
        "cxxModuleCMakeListsPath",
        "cxxModuleHeaderName",
    }
)

# Regexes used by the React Native CLI to find the Android package class and
# the module namespace (cli-platform-android findPackageClassName and
# getAndroidProject).
CLI_PACKAGE_CLASS_PATTERN = re.compile(
    r"class\s+(\w+[^(\s]*)[\s\w():]*(\s+implements\s+|:)[\s\w():,]*[^{]*ReactPackage"
)
CLI_NAMESPACE_PATTERN = re.compile(r"namespace\s*[=]*\s*[\"'](.+?)[\"']")

PODSPEC_EVALUATION_SCRIPT = r"""
require "json"
begin
  require "cocoapods-core"
rescue LoadError
  puts JSON.generate({ "skip" => "cocoapods-core is not installed" })
  exit 0
end

path = ARGV.fetch(0)
calls = nil
helpers = {
  install_modules_dependencies: proc { |spec|
    calls["install_modules_dependencies"] += 1
    spec.dependency("React-Core")
  },
  spm_dependency: proc { |spec, url:, requirement:, products:|
    calls["spm_dependency"] << {
      "url" => url,
      "requirement" => requirement.transform_keys(&:to_s),
      "products" => products,
    }
  },
  use_react_native!: proc { |*_args| nil },
}
modes = {
  "standalone" => [],
  "react_native" => %i[install_modules_dependencies spm_dependency use_react_native!],
  "react_native_without_spm" => %i[install_modules_dependencies use_react_native!],
}
results = {}
modes.each do |mode, names|
  calls = { "install_modules_dependencies" => 0, "spm_dependency" => [] }
  names.each do |name|
    Object.send(:define_method, name, &helpers.fetch(name))
    Object.send(:private, name)
  end
  begin
    spec = Pod::Specification.from_file(path)
    results[mode] = { "spec" => spec.to_hash, "calls" => calls }
  rescue StandardError => error
    results[mode] = { "error" => error.message }
  ensure
    names.each { |name| Object.send(:remove_method, name) }
  end
end
puts JSON.generate(results)
"""


def _run(command: list[str], *, cwd: Path = PACKAGE_DIR) -> str:
    env = {
        **os.environ,
        "LANG": "en_US.UTF-8",
        "LC_ALL": "en_US.UTF-8",
        "npm_config_audit": "false",
        "npm_config_fund": "false",
        "npm_config_update_notifier": "false",
    }
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


def _require_tool(name: str) -> str:
    tool = shutil.which(name)
    if tool is None:
        pytest.skip(f"{name} is required for this React Native packaging check")
    return tool


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _swift_bridge_selectors() -> set[str]:
    return set(re.findall(r"@objc\((\w+:(?:\w+:)*)\)", _read(SWIFT_MODULE)))


def _macro_arguments(source: str, macro: str) -> list[str]:
    """Return the balanced-parenthesis argument text of every macro use."""
    arguments = []
    for match in re.finditer(rf"\b{macro}\(", source):
        depth, start = 1, match.end()
        for index in range(start, len(source)):
            depth += {"(": 1, ")": -1}.get(source[index], 0)
            if depth == 0:
                arguments.append(source[start:index])
                break
    return arguments


def _export_file_selectors() -> set[str]:
    return {
        "".join(f"{label}:" for label in re.findall(r"(\w+):\(", body))
        for body in _macro_arguments(_read(EXPORT_FILE), "RCT_EXTERN_METHOD")
    }


def test_npm_pack_contains_native_sources_and_autolinking_config() -> None:
    npm = _require_tool("npm")

    output = _run([npm, "pack", "--dry-run", "--json", "--offline", "--ignore-scripts"])
    packed = {
        entry["path"].replace("\\", "/") for entry in json.loads(output)[0]["files"]
    }

    assert packed == EXPECTED_PACKED_FILES
    for path in packed:
        parts = set(path.lower().split("/"))
        assert not parts & {"test", "tests", "__tests__", "fixtures", "__fixtures__"}
        assert "__snapshots__" not in parts
        assert ".test." not in path and ".spec." not in path


def test_expected_tarball_covers_every_native_build_input() -> None:
    ios_sources = {
        path.relative_to(PACKAGE_DIR).as_posix()
        for path in IOS_DIR.rglob("*")
        if path.suffix in {".h", ".m", ".mm", ".swift"}
    }
    android_sources = {
        path.relative_to(PACKAGE_DIR).as_posix()
        for path in (ANDROID_DIR / "src" / "main").rglob("*")
        if path.is_file()
    }

    assert ios_sources == {"ios/OpenMedKitRN.m", "ios/OpenMedKitRN.swift"}
    assert ios_sources | android_sources <= EXPECTED_PACKED_FILES
    assert not list(ANDROID_DIR.glob("*.kt")), "Kotlin sources belong in src/main"


def test_ios_export_file_registers_the_swift_module_and_methods() -> None:
    export_file = _read(EXPORT_FILE)
    swift = _read(SWIFT_MODULE)

    assert "#import <React/RCTBridgeModule.h>" in export_file
    assert re.search(
        rf"@interface\s+RCT_EXTERN_MODULE\({MODULE_NAME},\s*NSObject\)", export_file
    )
    assert f"@objc({MODULE_NAME})" in swift
    assert re.search(
        rf'static func moduleName\(\) -> String! \{{\s*"{MODULE_NAME}"', swift
    )

    export_selectors = _export_file_selectors()
    assert export_selectors == _swift_bridge_selectors()
    assert {
        selector.split(":", 1)[0] for selector in export_selectors
    } == BRIDGE_METHODS
    for selector in export_selectors:
        assert selector.endswith("resolver:rejecter:")


def test_android_package_registers_the_module_for_autolinking() -> None:
    module = _read(KOTLIN_MODULE)
    package = _read(KOTLIN_PACKAGE)
    gradle = _read(BUILD_GRADLE)

    for source in (module, package):
        assert source.startswith(f"package {KOTLIN_PACKAGE_NAME}\n")
    match = CLI_PACKAGE_CLASS_PATTERN.search(package)
    assert match is not None and match.group(1) == "OpenMedKitRnPackage"
    assert KOTLIN_PACKAGE.name.endswith("Package.kt")
    assert "listOf(OpenMedKitRnModule(reactContext))" in package
    assert "ReactPackage" not in module

    assert f'const val NAME = "{MODULE_NAME}"' in module
    assert "@ReactModule(name = OpenMedKitRnModule.NAME)" in module
    assert "override fun getName(): String = NAME" in module
    assert set(re.findall(r"@ReactMethod\s+fun (\w+)\(", module)) == BRIDGE_METHODS

    namespace = CLI_NAMESPACE_PATTERN.search(gradle)
    assert namespace is not None and namespace.group(1) == KOTLIN_PACKAGE_NAME


def test_typescript_native_interface_matches_the_bridge_methods() -> None:
    source = _read(TS_ENTRY)
    interface = re.search(
        r"export interface OpenMedKitNativeModule \{(.*?)\n\}", source, re.S
    )

    assert interface is not None
    assert set(re.findall(r"^  (\w+)\(", interface.group(1), re.M)) == BRIDGE_METHODS
    assert f'OPENMED_REACT_NATIVE_MODULE_NAME = "{MODULE_NAME}"' in source
    assert "is not installed in the React Native runtime" in source


def test_android_gradle_module_pins_the_compile_baseline() -> None:
    gradle = _read(BUILD_GRADLE)
    package_json = json.loads(_read(PACKAGE_JSON))

    assert "if (project == rootProject)" in gradle
    assert 'classpath "com.android.tools.build:gradle:8.7.3"' in gradle
    assert 'classpath "org.jetbrains.kotlin:kotlin-gradle-plugin:2.0.21"' in gradle
    assert 'apply plugin: "com.android.library"' in gradle
    assert 'apply plugin: "kotlin-android"' in gradle
    assert 'compileSdk safeExtGet("compileSdkVersion", 35)' in gradle
    assert "minSdk 26" in gradle
    assert gradle.count("JavaVersion.VERSION_17") == 2
    assert 'jvmTarget = "17"' in gradle
    assert (
        "com.facebook.react:react-android:"
        f'${{safeExtGet("reactNativeVersion", "{REACT_NATIVE_BASELINE}")}}'
    ) in gradle
    assert f"com.github.maziyarpanahi:openmed:v{OPENMEDKIT_RELEASE}" in gradle
    assert 'includeGroup "com.github.maziyarpanahi"' in gradle
    assert "kotlinx-coroutines-core" in gradle

    baseline_minor = ".".join(REACT_NATIVE_BASELINE.split(".")[:2])
    assert package_json["peerDependencies"]["react-native"] == f">={baseline_minor}"


def test_podspec_declares_bridge_sources_and_the_openmedkit_swift_package() -> None:
    podspec = _read(PODSPEC)

    assert 's.name         = "openmedkit-react-native"' in podspec
    assert 'File.join(__dir__, "package.json")' in podspec
    assert 's.source_files = "ios/**/*.{h,m,mm,swift}"' in podspec
    assert 's.platforms    = { :ios => "17.0" }' in podspec
    assert "install_modules_dependencies(s)" in podspec
    assert 's.dependency "React-Core"' in podspec
    assert '"https://github.com/maziyarpanahi/openmed.git"' in podspec
    assert (
        f'openmedkit_swift_package_minimum_version = "{OPENMEDKIT_RELEASE}"' in podspec
    )
    assert 'products: ["OpenMedKit"]' in podspec


def test_podspec_is_valid_ruby() -> None:
    ruby = _require_tool("ruby")

    assert "Syntax OK" in _run([ruby, "-c", str(PODSPEC)])


def test_podspec_evaluates_under_cocoapods_with_and_without_react_native() -> None:
    ruby = _require_tool("ruby")

    results = json.loads(_run([ruby, "-e", PODSPEC_EVALUATION_SCRIPT, str(PODSPEC)]))
    if "skip" in results:
        pytest.skip(results["skip"])

    standalone = results["standalone"]["spec"]
    assert standalone["name"] == "openmedkit-react-native"
    assert standalone["version"] == json.loads(_read(PACKAGE_JSON))["version"]
    assert standalone["platforms"] == {"ios": "17.0"}
    assert standalone["source_files"] == "ios/**/*.{h,m,mm,swift}"
    assert standalone["dependencies"] == {"React-Core": []}
    assert standalone["pod_target_xcconfig"] == {"DEFINES_MODULE": "YES"}

    react_native = results["react_native"]
    assert react_native["calls"]["install_modules_dependencies"] == 1
    assert react_native["calls"]["spm_dependency"] == [
        {
            "url": "https://github.com/maziyarpanahi/openmed.git",
            "requirement": {
                "kind": "upToNextMajorVersion",
                "minimumVersion": OPENMEDKIT_RELEASE,
            },
            "products": ["OpenMedKit"],
        }
    ]

    assert "React Native 0.77 or later" in results["react_native_without_spm"]["error"]


def test_react_native_config_follows_the_cli_schema() -> None:
    node = _require_tool("node")
    package_json = json.loads(_read(PACKAGE_JSON))

    output = _run(
        [
            node,
            "-e",
            "process.stdout.write(JSON.stringify(require(process.argv[1])))",
            str(RN_CONFIG),
        ]
    )
    platforms = json.loads(output)["dependency"]["platforms"]

    assert "type" not in package_json, "the CLI require()s the CommonJS config"
    assert set(platforms["ios"]) <= CLI_IOS_DEPENDENCY_KEYS
    android = platforms["android"]
    assert set(android) <= CLI_ANDROID_DEPENDENCY_KEYS
    assert (PACKAGE_DIR / android["sourceDir"] / "build.gradle").is_file()
    assert android["packageImportPath"] == (
        f"import {KOTLIN_PACKAGE_NAME}.OpenMedKitRnPackage;"
    )
    assert android["packageInstance"] == "new OpenMedKitRnPackage()"


def test_native_rejections_and_logs_never_carry_caller_values() -> None:
    swift = _read(SWIFT_MODULE)
    kotlin = _read(KOTLIN_MODULE)

    assert "localizedDescription" not in swift
    assert (
        len(re.findall(r'Self\.rejectValueFree\(reject, code: "openmedkit_\w+"', swift))
        == 4
    )
    assert swift.count("reject(code, message, nil)") == 1
    assert len(re.findall(r"\breject\(", swift)) == 1

    assert "error.message" not in kotlin
    assert kotlin.count("promise.reject(") == 4
    assert (
        len(
            re.findall(
                r'promise\.reject\("openmedkit_\w+", valueFreeMessage\(error\)\)',
                kotlin,
            )
        )
        == 4
    )

    swift_messages = re.findall(
        r'return "([^"]*)"', swift.split("private enum BridgeError")[1]
    )
    kotlin_messages = re.findall(r'BridgeException\("([^"]*)"\)', kotlin)
    assert swift_messages and kotlin_messages
    for message in swift_messages:
        assert set(re.findall(r"\\\((\w+)\)", message)) <= {"key"}
    for message in kotlin_messages:
        assert set(re.findall(r"\$\{?(\w+)", message)) <= {"key"}

    for source in (swift, _read(EXPORT_FILE), kotlin, _read(KOTLIN_PACKAGE)):
        for call in (
            "print(",
            "NSLog(",
            "os_log(",
            "Log.",
            "println(",
            "printStackTrace",
        ):
            assert call not in source
