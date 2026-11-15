"""Policy tests for the R8-minified OpenMedKit consumer app and its CI wiring."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from scripts.android.check_r8_consumer import (
    dontwarn_patterns,
    read_allowlist,
    read_properties,
    read_rules,
)

ROOT = Path(__file__).parents[3]
ANDROID = ROOT / "android"
MODULE = ANDROID / "consumer-r8-smoke"
BUILD = MODULE / "build.gradle.kts"
LIBRARY_BUILD = ANDROID / "openmedkit" / "build.gradle.kts"
CONSUMER_RULES = ANDROID / "openmedkit" / "consumer-rules.pro"
BUDGETS = ANDROID / "gradle" / "budgets.properties"
WORKFLOW = ROOT / ".github" / "workflows" / "android-ci.yml"
SPLIT_ABIS = ["arm64-v8a", "armeabi-v7a", "x86", "x86_64"]


def _build() -> str:
    return BUILD.read_text(encoding="utf-8")


def _string_list(build: str, name: str) -> list[str]:
    match = re.search(rf"val {name} = listOf\((.*?)\)\s*$", build, re.DOTALL | re.M)
    assert match is not None, name
    return re.findall(r'"([^"]+)"', match.group(1))


def test_module_is_included_and_minifies_with_r8_full_mode() -> None:
    build = _build()
    settings = (ANDROID / "settings.gradle.kts").read_text(encoding="utf-8")

    assert 'include(":consumer-r8-smoke")' in settings
    assert 'id("com.android.application")' in build
    assert "isMinifyEnabled = true" in build
    assert 'getDefaultProguardFile("proguard-android-optimize.txt")' in build
    assert '"proguard-rules.pro"' in build
    # Full mode is the AGP 8 default; the module refuses to configure without it.
    assert 'gradleProperty("android.enableR8.fullMode").getOrElse("true")' in build
    assert "android.enableR8.fullMode=false" not in (
        ANDROID / "gradle.properties"
    ).read_text(encoding="utf-8")
    assert (
        'beforeVariants(selector().withBuildType("debug")) { it.enable = false }'
        in (build)
    )


def test_consumer_adds_no_rules_of_its_own() -> None:
    rules = read_rules((MODULE / "proguard-rules.pro").read_text(encoding="utf-8"))

    assert rules == []


def test_flavors_cover_the_aar_a_baseline_and_a_negative_control() -> None:
    build = _build()

    for flavor in ("openmedkit", "baseline", "missingKeepRule"):
        assert f'create("{flavor}")' in build
    assert '"openmedkitImplementation"(openmedkitAarFiles)' in build
    assert '"missingKeepRuleImplementation"(files(missingKeepRuleAar)' in build
    assert 'builtBy(":openmedkit:bundleReleaseAar")' in build
    assert 'kotlin.srcDir("src/openmedkit/kotlin")' in build
    baseline = (MODULE / "src" / "baseline" / "kotlin").rglob("*.kt")
    assert all("com.openmed" not in path.read_text() for path in baseline)


def test_negative_control_drops_a_rule_the_aar_really_ships() -> None:
    build = _build()
    rule = re.search(r'val negativeControlRule = "([^"]+)"', build)
    expectation = re.search(r'val negativeControlExpectation = "([^"]+)"', build)

    assert rule is not None and expectation is not None
    assert rule.group(1) in read_rules(CONSUMER_RULES.read_text(encoding="utf-8"))
    assert expectation.group(1) == "jni_class_not_kept:ai.onnxruntime."
    assert '"verifyMissingKeepRuleNegativeControl"' in build
    assert "listOf(negativeControlExpectation)" in build


def test_consumer_rules_keep_every_class_bundled_jni_libraries_name() -> None:
    rules = read_rules(CONSUMER_RULES.read_text(encoding="utf-8"))

    assert "-keep class ai.onnxruntime.** { *; }" in rules
    assert "-keep class ai.djl.huggingface.tokenizers.jni.** { *; }" in rules
    # libdjl_tokenizer.so throws it by name; R8 renamed it before this rule.
    assert "-keep class ai.djl.engine.EngineException { *; }" in rules
    assert dontwarn_patterns(rules) == []
    assert "-ignorewarnings" not in rules


def test_runtime_dependencies_match_the_openmedkit_pom() -> None:
    library = LIBRARY_BUILD.read_text(encoding="utf-8")
    dependencies = library[library.index("\ndependencies {") :]
    expected = re.findall(r"^\s+implementation\((libs\.[\w.]+)\)", dependencies, re.M)

    consumer = re.search(
        r"val openmedkitRuntimeDependencies = listOf\((.*?)\n\)", _build(), re.DOTALL
    )
    assert consumer is not None
    assert re.findall(r"(libs\.[\w.]+)", consumer.group(1)) == expected
    assert len(expected) == 6


def test_desktop_jni_resources_are_excluded_and_documented() -> None:
    excludes = _string_list(_build(), "desktopNativeResourceExcludes")
    readme = (ANDROID / "README.md").read_text(encoding="utf-8")
    quickstart = (ROOT / "docs" / "android-quickstart.md").read_text(encoding="utf-8")

    assert excludes == ["native/lib/**", "com/sun/jna/*/*jnidispatch*"]
    assert "excludes += desktopNativeResourceExcludes" in _build()
    for document in (readme, quickstart):
        for pattern in excludes:
            assert f'"{pattern}"' in document


def test_abi_splits_match_the_committed_size_budgets() -> None:
    build = _build()
    budgets = read_properties(BUDGETS.read_text(encoding="utf-8"))

    assert _string_list(build, "splitAbis") == SPLIT_ABIS
    assert "isUniversalApk = false" in build
    consumer_keys = sorted(key for key in budgets if key.startswith("consumerApk."))
    assert consumer_keys == sorted(
        f"consumerApk.{abi}.maxDeltaBytes" for abi in SPLIT_ABIS
    )
    assert all(
        int(budgets[key]) > int(budgets["aar.maxBytes"]) for key in consumer_keys
    )


def test_gate_requires_openmedkit_jni_libraries_and_packages() -> None:
    build = _build()

    assert _string_list(build, "requiredNativeLibraries") == [
        "onnxruntime",
        "onnxruntime4j_jni",
        "djl_tokenizer",
    ]
    assert _string_list(build, "jniPackages") == [
        "ai/onnxruntime",
        "ai/djl",
        "com/openmed",
    ]
    for task in (
        '"verifyOpenmedkitR8Gate"',
        "verifyConsumerApkSize",
        '"verifyOpenmedkitReleaseApkZipAlignment"',
        '"testOpenmedkitReleaseUnitTest"',
    ):
        assert task in build


def test_smoke_app_cannot_reach_the_network() -> None:
    manifest = (MODULE / "src" / "main" / "AndroidManifest.xml").read_text()

    assert re.search(
        r'android:name="android\.permission\.INTERNET"\s+tools:node="remove"', manifest
    )


def test_dontwarn_allowlist_is_reviewed_and_narrow() -> None:
    allowlist = read_allowlist(
        (MODULE / "r8-dontwarn-allowlist.txt").read_text(encoding="utf-8")
    )

    assert allowlist
    assert not any(
        entry.startswith(("ai.onnxruntime", "ai.djl")) for entry in allowlist
    )


def test_android_ci_runs_the_r8_consumer_gate_and_publishes_sizes() -> None:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    steps = {
        step.get("name"): step for step in workflow["jobs"]["build-and-test"]["steps"]
    }
    triggers = workflow[True] if True in workflow else workflow["on"]

    gate = steps["Verify R8-minified OpenMedKit consumer"]
    assert gate["working-directory"] == "android"
    assert ":consumer-r8-smoke:verifyR8Consumer" in gate["run"]
    for event in ("push", "pull_request"):
        for script in ("check_r8_consumer.py", "generate_r8_smoke_model.py"):
            assert f"scripts/android/{script}" in triggers[event]["paths"]
    summary = steps["Publish Android budget summary"]["run"]
    assert "consumer-apk-size.properties" in summary
    assert "-r8-gate.properties" in summary
    assert "consumerApk.${abi}.maxDeltaBytes" in summary
    uploads = steps["Upload unit test report"]["with"]["path"]
    assert "android/consumer-r8-smoke/build/reports/" in uploads
