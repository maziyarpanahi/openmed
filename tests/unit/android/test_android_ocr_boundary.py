"""Policy tests for the Android core/ML Kit OCR dependency boundary."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).parents[3]
ANDROID = ROOT / "android"
CORE = ANDROID / "openmedkit"
MLKIT = ANDROID / "openmedkit-mlkit"
SCAN_DEMO = ANDROID / "OpenMedScanDemo" / "app"
BOUNDARY_SCRIPT = ANDROID / "gradle" / "dependency-boundary.gradle.kts"
CI_WORKFLOW = ROOT / ".github" / "workflows" / "android-ci.yml"
PUBLISH_WORKFLOW = ROOT / ".github" / "workflows" / "android-publish.yml"
JITPACK_CONFIG = ROOT / "jitpack.yml"
INTEGRATION_GUIDE = ROOT / "docs" / "android-integration.md"

VENDOR_PACKAGES = (
    "com.google.mlkit",
    "com.google.android.gms",
    "com.google.android.datatransport",
    "com.google.firebase",
)


def _kotlin_sources(module: Path, source_set: str) -> list[Path]:
    return sorted((module / "src" / source_set).rglob("*.kt"))


def test_settings_include_core_and_optional_mlkit_modules() -> None:
    settings = (ANDROID / "settings.gradle.kts").read_text(encoding="utf-8")

    assert 'include(":openmedkit")' in settings
    assert 'include(":openmedkit-mlkit")' in settings


def test_core_module_declares_and_imports_no_vendor_ocr_stack() -> None:
    build = (CORE / "build.gradle.kts").read_text(encoding="utf-8")

    assert "libs.mlkit" not in build
    assert not any(f'"{package}' in build for package in VENDOR_PACKAGES)
    assert not (
        CORE / "src/main/kotlin/com/openmed/openmedkit/ocr/MlKitOcrAdapter.kt"
    ).exists()
    for source_set in ("main", "test"):
        for source in _kotlin_sources(CORE, source_set):
            text = source.read_text(encoding="utf-8")
            imports = re.findall(r"^import\s+(\S+)", text, flags=re.MULTILINE)
            vendor_imports = [
                name for name in imports if name.startswith(VENDOR_PACKAGES)
            ]
            assert vendor_imports == [], source.relative_to(ROOT)


def test_core_and_optional_module_run_the_shared_boundary_scan() -> None:
    core_build = (CORE / "build.gradle.kts").read_text(encoding="utf-8")
    mlkit_build = (MLKIT / "build.gradle.kts").read_text(encoding="utf-8")
    apply_line = (
        'apply(from = rootProject.file("gradle/dependency-boundary.gradle.kts"))'
    )

    assert 'extra["openmedkitDependencyBoundary"] = "core"' in core_build
    assert apply_line in core_build
    assert 'dependsOn("verifyDependencyBoundary")' in core_build
    assert 'extra["openmedkitDependencyBoundary"] = "optional-vendor"' in mlkit_build
    assert apply_line in mlkit_build
    assert 'dependsOn("verifyDependencyBoundary")' in mlkit_build


def test_boundary_scan_covers_pom_classpath_and_consumer_manifests() -> None:
    script = BOUNDARY_SCRIPT.read_text(encoding="utf-8")

    for group in VENDOR_PACKAGES:
        assert f'"{group}"' in script
    for permission in ("INTERNET", "ACCESS_NETWORK_STATE", "ACCESS_WIFI_STATE"):
        assert f'"android.permission.{permission}"' in script
    assert 'setOf("provider", "service", "receiver")' in script
    assert '"releaseRuntimeClasspath"' in script
    assert 'dependsOn("generatePomFileForReleasePublication")' in script
    assert '"android-manifest"' in script
    assert "read no dependency" in script
    assert 'listOf("classpath:", "pom:", "permission:", "component:")' in script
    assert "disallow-doctype-decl" in script


def test_optional_module_owns_the_mlkit_dependency_and_publication() -> None:
    build = (MLKIT / "build.gradle.kts").read_text(encoding="utf-8")
    adapter = (
        MLKIT / "src/main/kotlin/com/openmed/openmedkit/ocr/MlKitOcrAdapter.kt"
    ).read_text(encoding="utf-8")

    assert 'api(project(":openmedkit"))' in build
    assert "api(libs.mlkit.text.recognition)" in build
    assert 'gradleProperty("openmedAndroidMlKitArtifact")' in build
    assert '.orElse("$coreArtifact-mlkit")' in build
    assert 'name.set("OpenMedKit Android ML Kit OCR")' in build
    assert "override suspend fun recognize(image: OcrImage): OcrResult" in adapter
    assert "suspend fun recognize(image: InputImage): OcrResult" in adapter


def test_scan_demo_uses_optional_module_and_neutral_images() -> None:
    build = (SCAN_DEMO / "build.gradle.kts").read_text(encoding="utf-8")
    sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((SCAN_DEMO / "src").rglob("*.kt"))
    )

    assert 'implementation(project(":openmedkit-mlkit"))' in build
    assert "com.google.mlkit" not in build
    assert "com.google.mlkit" not in sources
    assert "OcrImage.fromEncodedBytes" in sources


def test_ci_publish_and_jitpack_cover_the_optional_module() -> None:
    ci = CI_WORKFLOW.read_text(encoding="utf-8")
    publish = PUBLISH_WORKFLOW.read_text(encoding="utf-8")
    jitpack = yaml.safe_load(JITPACK_CONFIG.read_text(encoding="utf-8"))
    install = jitpack["install"][0]

    assert ":openmedkit:verifyDependencyBoundary" in ci
    assert ":openmedkit-mlkit:verifyDependencyBoundary" in ci
    assert ":openmedkit-mlkit:testDebugUnitTest" in ci
    assert ":openmedkit-mlkit:testDebugUnitTest" in publish
    assert ":openmedkit:verifyDependencyBoundary" in publish
    assert ":openmedkit-mlkit:bundleCentralPortalPublication" in publish
    assert "for artifact in openmedkit openmedkit-mlkit; do" in publish
    assert "openmedkit-mlkit-${{ needs.guard.outputs.version }}.zip" in publish
    assert ":openmedkit:publishReleasePublicationToMavenLocal" in install
    assert ":openmedkit-mlkit:publishReleasePublicationToMavenLocal" in install


def test_integration_guide_documents_migration_license_and_telemetry() -> None:
    guide = INTEGRATION_GUIDE.read_text(encoding="utf-8")

    assert "### Optional ML Kit OCR module" in guide
    assert "### Migrating from the bundled ML Kit adapter" in guide
    assert "**Breaking:**" in guide
    assert "Google's ML Kit terms" in guide
    assert "android.permission.INTERNET" in guide
    assert "com.google.android.datatransport" in guide
    assert "OcrImage.fromBitmap" in guide
    assert "OcrImage.fromEncodedBytes" in guide
    assert "com.github.maziyarpanahi:openmed-mlkit:" in guide
    assert "verifyDependencyBoundary" in guide
