"""Android build toolchain policy tests (AGP 8, JDK 17, SDK levels, 16 KB gate)."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).parents[3]
ANDROID = ROOT / "android"
CATALOG = ANDROID / "gradle" / "libs.versions.toml"
WRAPPER = ANDROID / "gradle" / "wrapper" / "gradle-wrapper.properties"
ROOT_BUILD = ANDROID / "build.gradle.kts"
SETTINGS = ANDROID / "settings.gradle.kts"
LIBRARY_BUILD = ANDROID / "openmedkit" / "build.gradle.kts"
DEMO_BUILDS = {
    name: ANDROID / name / "app" / "build.gradle.kts"
    for name in ("OpenMedDemo", "OpenMedScanDemo", "OpenMedMapleDemo")
}
ANDROID_CI = ROOT / ".github" / "workflows" / "android-ci.yml"
ANDROID_PUBLISH = ROOT / ".github" / "workflows" / "android-publish.yml"
JITPACK = ROOT / "jitpack.yml"
ANDROID_README = ANDROID / "README.md"
ALIGNMENT_SCRIPT = "scripts/android/check_apk_zip_alignment.py"

MIN_AGP_FOR_16_KB_ZIP_ALIGNMENT = (8, 5, 1)
MIN_COMPILE_SDK = 35
PLAY_TARGET_SDK = 35
LIBRARY_MIN_SDK = 26


def _version(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", value)[:3])


def _catalog_versions() -> dict[str, str]:
    return tomllib.loads(CATALOG.read_text(encoding="utf-8"))["versions"]


def _properties(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            key, value = line.split("=", maxsplit=1)
            values[key] = value
    return values


def _int_setting(build: str, name: str) -> list[int]:
    return [int(value) for value in re.findall(rf"\b{name} = (\d+)", build)]


def _block(build: str, header: str) -> str:
    start = build.index(header)
    depth = 0
    for index in range(build.index("{", start), len(build)):
        depth += {"{": 1, "}": -1}.get(build[index], 0)
        if depth == 0:
            return build[start : index + 1]
    raise AssertionError(f"unterminated block {header!r}")


def _gradle_version() -> str:
    match = re.search(
        r"gradle-(\d+\.\d+(?:\.\d+)?)-bin\.zip", _properties(WRAPPER)["distributionUrl"]
    )
    assert match is not None
    return match.group(1)


def test_catalog_pins_agp_8_with_16_kb_zip_alignment_and_kotlin_2() -> None:
    versions = _catalog_versions()

    assert _version(versions["agp"]) >= MIN_AGP_FOR_16_KB_ZIP_ALIGNMENT
    assert _version(versions["agp"])[0] == 8
    assert _version(versions["kotlin"])[0] == 2
    plugins = tomllib.loads(CATALOG.read_text(encoding="utf-8"))["plugins"]
    assert plugins["kotlin-compose"] == {
        "id": "org.jetbrains.kotlin.plugin.compose",
        "version": {"ref": "kotlin"},
    }
    assert "org.jetbrains.kotlin.plugin.compose" in SETTINGS.read_text(encoding="utf-8")


def test_gradle_wrapper_is_8_x_with_a_pinned_distribution_checksum() -> None:
    properties = _properties(WRAPPER)

    assert _version(_gradle_version())[0] == 8
    assert properties["distributionUrl"].startswith(
        r"https\://services.gradle.org/distributions/"
    )
    assert re.fullmatch(r"[0-9a-f]{64}", properties["distributionSha256Sum"])
    assert properties["validateDistributionUrl"] == "true"


def test_library_compiles_against_current_sdk_and_keeps_its_consumer_floor() -> None:
    build = LIBRARY_BUILD.read_text(encoding="utf-8")
    default_config = _block(build, "defaultConfig {")

    assert 'namespace = "org.openmed.openmedkit"' in build
    assert all(level >= MIN_COMPILE_SDK for level in _int_setting(build, "compileSdk"))
    assert _int_setting(default_config, "minSdk") == [LIBRARY_MIN_SDK]
    # AGP 8 deprecates targetSdk in a library defaultConfig.
    assert "targetSdk" not in default_config
    assert _int_setting(_block(build, "testOptions {"), "targetSdk") == [
        PLAY_TARGET_SDK
    ]
    assert _int_setting(_block(build, "lint {"), "targetSdk") == [PLAY_TARGET_SDK]
    # The build runs on JDK 17 while consumers still receive Java 11 bytecode.
    assert "jvmToolchain(17)" in build
    assert "sourceCompatibility = JavaVersion.VERSION_11" in build
    assert "targetCompatibility = JavaVersion.VERSION_11" in build
    assert "jvmTarget.set(JvmTarget.JVM_11)" in build


def test_demos_target_the_play_required_sdk_level() -> None:
    for name, path in DEMO_BUILDS.items():
        build = path.read_text(encoding="utf-8")

        assert re.search(r'namespace = "org\.openmed\.\w+"', build), name
        assert _int_setting(build, "compileSdk") == [MIN_COMPILE_SDK], name
        assert _int_setting(build, "targetSdk") == [PLAY_TARGET_SDK], name
        assert _int_setting(build, "minSdk") == [LIBRARY_MIN_SDK], name
        assert 'id("org.jetbrains.kotlin.plugin.compose")' in build, name
        assert "jvmTarget.set(JvmTarget.JVM_11)" in build, name


def test_android_builds_use_no_deprecated_agp_7_dsl_or_dex_workarounds() -> None:
    builds = [LIBRARY_BUILD, *DEMO_BUILDS.values()]

    for path in builds:
        build = path.read_text(encoding="utf-8")
        assert "kotlinOptions {" not in build, path
        assert "kotlinCompilerExtensionVersion" not in build, path
        assert "packagingOptions" not in build, path
        assert "android-compatible" not in build, path
    assert "android-compatible" not in CATALOG.read_text(encoding="utf-8")


def test_root_build_registers_16_kb_zip_alignment_gates() -> None:
    build = ROOT_BUILD.read_text(encoding="utf-8")

    assert ALIGNMENT_SCRIPT in build
    assert '"--page-size-kb",\n            "16",' in build
    assert '"--require-native-libraries"' in build
    assert "variant.artifacts.get(SingleArtifact.APK)" in build
    assert "testComponent.artifacts.get(SingleArtifact.APK)" in build
    assert 'pluginManager.withPlugin("com.android.application")' in build
    assert 'pluginManager.withPlugin("com.android.library")' in build
    assert "val verifyApkZipAlignment by tasks.registering" in build
    assert (ROOT / ALIGNMENT_SCRIPT).is_file()


def test_android_ci_uses_jdk_17_and_gates_demo_apk_alignment() -> None:
    workflow = yaml.safe_load(ANDROID_CI.read_text(encoding="utf-8"))
    steps = {
        step.get("name"): step for step in workflow["jobs"]["build-and-test"]["steps"]
    }
    triggers = workflow[True] if True in workflow else workflow["on"]

    assert steps["Set up Java"]["with"]["java-version"] == "17"
    for event in ("push", "pull_request"):
        assert ALIGNMENT_SCRIPT in triggers[event]["paths"]
    assemble = steps["Assemble demo APKs"]["run"]
    for name in DEMO_BUILDS:
        assert f":{name}:app:assembleDebug" in assemble
    assert (
        "./gradlew verifyApkZipAlignment"
        in steps["Verify 16 KB APK zip alignment"]["run"]
    )
    cross_check = steps["Cross-check APK alignment with SDK zipalign"]["run"]
    assert '"$zipalign" -c -P 16 4 "$apk"' in cross_check
    assert "openmedkit/build/outputs/apk/androidTest/debug/*.apk" in cross_check
    assert "-ne 4 ]]" in cross_check
    assert (
        ":openmedkit:verifyAndroidBudgets"
        in steps["Verify Android library budgets"]["run"]
    )


def test_publish_and_jitpack_builds_use_jdk_17() -> None:
    publish = yaml.safe_load(ANDROID_PUBLISH.read_text(encoding="utf-8"))
    java_steps = [
        step
        for step in publish["jobs"]["build"]["steps"]
        if step.get("name") == "Set up Java"
    ]

    assert [step["with"]["java-version"] for step in java_steps] == ["17"]
    assert yaml.safe_load(JITPACK.read_text(encoding="utf-8"))["jdk"] == ["openjdk17"]


def test_readme_documents_the_checked_in_toolchain() -> None:
    readme = ANDROID_README.read_text(encoding="utf-8")
    section = readme[readme.index("## Toolchain Requirements") :]
    section = section[: section.index("\n## ", 1)]
    versions = _catalog_versions()

    assert "| Build JDK | 17" in section
    assert f"| Gradle | {_gradle_version()} " in section
    assert f"| Android Gradle Plugin | {versions['agp']} " in section
    assert f"| Kotlin | {versions['kotlin']}," in section
    assert f"Platform {MIN_COMPILE_SDK} (`compileSdk {MIN_COMPILE_SDK}`)" in section
    assert f"| Library `minSdk` | {LIBRARY_MIN_SDK} |" in section
    assert f"| Demo `targetSdk` | {PLAY_TARGET_SDK}," in section
    assert "./gradlew verifyApkZipAlignment" in readme
    assert "Install JDK 17 and Android SDK Platform 35" in readme
