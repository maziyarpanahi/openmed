"""Gating checks for the React Native bridge parity workflow and example app."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "react-native-parity.yml"
PACKAGE = ROOT / "js" / "openmedkit-react-native"
EXAMPLE = PACKAGE / "example"
CHECK_PACKAGING = EXAMPLE / "scripts" / "check-packaging.sh"
PACKAGING_FILES = (
    "openmedkit-react-native.podspec",
    "ios/OpenMedKitRN.m",
    "android/build.gradle",
    "android/src/main/java/OpenMedKitRnPackage.kt",
    "react-native.config.js",
)


def _workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _commands(job: dict) -> str:
    return "\n".join(
        str(step.get("run", "")) + str(step.get("with", {}).get("script", ""))
        for step in job["steps"]
    )


def test_harness_job_runs_the_bridge_and_harness_tests_on_pull_requests() -> None:
    workflow = _workflow()
    triggers = workflow.get("on", workflow.get(True))
    assert {"pull_request", "push", "workflow_dispatch"} <= set(triggers)
    assert workflow["permissions"] == {"contents": "read"}

    harness = workflow["jobs"]["harness"]
    assert "if" not in harness
    commands = _commands(harness)
    assert "npm ci" in commands
    assert "npm test" in commands


def test_native_jobs_are_dispatch_only_until_the_packaging_gate_passes() -> None:
    jobs = _workflow()["jobs"]
    packaging = jobs["packaging"]
    assert packaging["if"] == "github.event_name == 'workflow_dispatch'"
    assert "check-packaging.sh" in _commands(packaging)

    for name, platform in (("ios-simulator", "ios"), ("android-emulator", "android")):
        job = jobs[name]
        assert job["needs"] == "packaging"
        assert job["if"].startswith("github.event_name == 'workflow_dispatch'")
        commands = _commands(job)
        assert f"prepare-native-app.sh {platform}" in commands
        assert f"run-on-device.sh {platform}" in commands
        assert f"--platform {platform}" in commands
        assert "check-report.ts" in commands
        assert "stage-model-fixture.ts verify" in commands

    assert "-configuration Release" in _commands(jobs["ios-simulator"])
    assert "assembleRelease" in _commands(jobs["android-emulator"])


def test_device_runner_stages_fixtures_locally_and_denies_android_network() -> None:
    script = (EXAMPLE / "scripts" / "run-on-device.sh").read_text(encoding="utf-8")
    assert "airplane-mode enable" in script
    assert "svc wifi disable" in script
    assert "huggingface" not in script
    assert "curl" not in script and "wget" not in script


def test_example_pins_react_native_077_or_newer_with_a_lockfile() -> None:
    package = json.loads((EXAMPLE / "package.json").read_text(encoding="utf-8"))
    version = package["dependencies"]["react-native"]
    assert re.fullmatch(r"\d+\.\d+\.\d+", version), "pin an exact version"
    major, minor, _ = (int(part) for part in version.split("."))
    assert (major, minor) >= (0, 77)
    assert package["dependencies"]["@openmed/openmedkit-react-native"] == "file:.."
    assert package["dependencies"]["openmedkit-parity-probe"] == "file:./probe"

    lock = json.loads((EXAMPLE / "package-lock.json").read_text(encoding="utf-8"))
    assert lock["packages"]["node_modules/react-native"]["version"] == version


def _packaging_tree(tmp_path: Path, *, complete: bool, module_type: bool) -> Path:
    package = tmp_path / "pkg"
    scripts = package / "example" / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy2(CHECK_PACKAGING, scripts / CHECK_PACKAGING.name)
    manifest = {"name": "@openmed/openmedkit-react-native"}
    if module_type:
        manifest["type"] = "module"
    (package / "package.json").write_text(json.dumps(manifest), encoding="utf-8")
    (package / "android").mkdir()
    if complete:
        for relative in PACKAGING_FILES:
            target = package / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("", encoding="utf-8")
    return scripts / CHECK_PACKAGING.name


@pytest.mark.parametrize(
    ("complete", "module_type", "expected"),
    [(False, True, 1), (True, True, 1), (True, False, 0)],
)
def test_packaging_check_fails_closed_without_the_4478_packaging(
    tmp_path: Path, complete: bool, module_type: bool, expected: int
) -> None:
    if shutil.which("bash") is None or shutil.which("node") is None:
        pytest.skip("bash and node are required for the packaging check")
    script = _packaging_tree(tmp_path, complete=complete, module_type=module_type)

    completed = subprocess.run(
        ["bash", str(script)],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    assert completed.returncode == expected, completed.stdout + completed.stderr
    if expected:
        assert "#4478" in completed.stderr
