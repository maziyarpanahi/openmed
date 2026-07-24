"""Android ICU parity dependency and workflow policy tests."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parents[3]
VERSION_CATALOG = ROOT / "android" / "gradle" / "libs.versions.toml"
BUILD_FILE = ROOT / "android" / "openmedkit" / "build.gradle.kts"
README = ROOT / "android" / "openmedkit" / "README.md"
WORKFLOW = ROOT / ".github" / "workflows" / "android-ci.yml"


def test_icu4j_is_test_only_and_records_permissive_unicode_license() -> None:
    catalog = VERSION_CATALOG.read_text(encoding="utf-8")
    build = BUILD_FILE.read_text(encoding="utf-8")
    readme = README.read_text(encoding="utf-8")
    icu_section = readme.split("## ICU Boundary Segmentation", maxsplit=1)[1]

    assert 'icu4j = "78.3"' in catalog
    assert 'module = "com.ibm.icu:icu4j"' in catalog
    assert "testImplementation(libs.icu4j)" in build
    assert "\n    implementation(libs.icu4j)" not in build
    assert "Unicode-3.0" in icu_section
    assert "ICU/Unicode License" in icu_section
    assert "https://www.unicode.org/license.txt" in icu_section
    assert not any(
        marker in icu_section
        for marker in ("AGPL", "GPL-", "LGPL", "SSPL", "source-available")
    )


def test_android_ci_runs_offset_and_segmentation_parity_suites() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "tests/fixtures/parity/offset_contract.json" in workflow
    assert "tests/fixtures/processing/zh_segmentation_gold.json" in workflow
    assert "--tests '*OffsetContractParityTest'" in workflow
    assert "--tests '*IcuSegmentationFallbackTest'" in workflow
