"""Android AAR-size and cold-start budget gate policy tests."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parents[3]
BUDGETS = ROOT / "android" / "gradle" / "budgets.properties"
BUILD_FILE = ROOT / "android" / "openmedkit" / "build.gradle.kts"
COLD_START_TEST = (
    ROOT
    / "android"
    / "openmedkit"
    / "src"
    / "test"
    / "kotlin"
    / "com"
    / "openmed"
    / "openmedkit"
    / "ColdStartBudgetTest.kt"
)
WORKFLOW = ROOT / ".github" / "workflows" / "android-ci.yml"


def _read_budgets() -> dict[str, int]:
    values: dict[str, int] = {}
    for raw_line in BUDGETS.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, raw_value = line.split("=", maxsplit=1)
        values[key] = int(raw_value)
    return values


def test_android_budgets_are_committed_positive_ceiling_values() -> None:
    budgets = _read_budgets()

    assert budgets.keys() == {"aar.maxBytes", "coldStart.maxMillis"}
    assert all(value > 0 for value in budgets.values())


def test_gradle_wires_both_measurements_into_one_verification_gate() -> None:
    build = BUILD_FILE.read_text(encoding="utf-8")

    assert 'readPositiveBudget("aar.maxBytes")' in build
    assert 'readPositiveBudget("coldStart.maxMillis")' in build
    assert 'tasks.register("verifyAndroidBudgets")' in build
    assert 'dependsOn(verifyReleaseAarSize, "testDebugUnitTest")' in build
    assert "release-aar-size.properties" in build
    assert "${name}-cold-start.properties" in build


def test_cold_start_measurement_is_robolectric_and_network_rejecting() -> None:
    source = COLD_START_TEST.read_text(encoding="utf-8")

    assert "RobolectricTestRunner::class" in source
    assert "ModelCatalog.load(context)" in source
    assert "ModelCache(cacheDirectory)" in source
    assert "ModelDownloader(cache, NetworkRejectingClient)" in source
    assert "Cold-start initialization must not access the network" in source
    assert source.index("measurementFile.writeText") < source.index(
        '"OpenMedKit cold start took $elapsedMillis ms'
    )


def test_android_ci_runs_gate_and_always_publishes_measurements() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert ":openmedkit:verifyAndroidBudgets --continue --stacktrace" in workflow
    assert "if: always()" in workflow
    assert 'echo "### Android library budgets"' in workflow
    assert "release-aar-size.properties" in workflow
    assert "testDebugUnitTest-cold-start.properties" in workflow
    assert "android/openmedkit/build/reports/budgets/" in workflow
