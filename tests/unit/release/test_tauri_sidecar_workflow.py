"""Lane definition checks for the Tauri sidecar qualification workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / ".github" / "workflows" / "tauri-sidecar.yml"
CRATE = ROOT / "js" / "openmedkit-tauri" / "src-tauri"
DECLARED_PLATFORMS = {"ubuntu-24.04", "macos-15", "windows-2022"}


def _workflow() -> dict[str, Any]:
    loaded = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _triggers(workflow: dict[str, Any]) -> dict[str, Any]:
    # PyYAML reads the bare ``on`` key as boolean True.
    triggers = workflow.get("on", workflow.get(True))
    assert isinstance(triggers, dict)
    return triggers


def _run_text(job: dict[str, Any]) -> str:
    return "\n".join(str(step.get("run", "")) for step in job["steps"])


def test_fake_process_lane_runs_by_default_on_every_declared_platform() -> None:
    workflow = _workflow()
    triggers = _triggers(workflow)
    assert {"push", "pull_request", "workflow_dispatch"} <= set(triggers)
    assert "schedule" not in triggers

    job = workflow["jobs"]["fake-process"]
    assert "if" not in job
    assert set(job["strategy"]["matrix"]["os"]) == DECLARED_PLATFORMS
    commands = _run_text(job)
    assert "cargo test --locked" in commands
    assert "--ignored" not in commands


def test_packaged_lane_is_opt_in_and_never_runs_by_default() -> None:
    workflow = _workflow()
    dispatch = _triggers(workflow)["workflow_dispatch"]
    lane_input = dispatch["inputs"]["packaged_lane"]
    assert lane_input["type"] == "boolean"
    assert lane_input["default"] is False

    job = workflow["jobs"]["packaged-sidecar"]
    condition = job["if"]
    assert "github.event_name == 'workflow_dispatch'" in condition
    assert "inputs.packaged_lane" in condition
    assert set(job["strategy"]["matrix"]["os"]) == DECLARED_PLATFORMS

    commands = _run_text(job)
    assert "pyinstaller" in commands
    assert "openmed-sidecar-manifest" in commands
    assert "--test packaged_sidecar_lane -- --ignored" in commands


def test_lane_test_is_ignored_unless_requested() -> None:
    source = (CRATE / "tests" / "packaged_sidecar_lane.rs").read_text(encoding="utf-8")
    assert "#[ignore" in source
    assert "OPENMED_TAURI_LANE_SIDECAR" in source


def test_workflow_is_read_only_and_does_not_persist_credentials() -> None:
    workflow = _workflow()
    assert workflow["permissions"] == {"contents": "read"}
    for job in workflow["jobs"].values():
        checkouts = [
            step
            for step in job["steps"]
            if str(step.get("uses", "")).startswith("actions/checkout@")
        ]
        assert checkouts
        assert all(step["with"]["persist-credentials"] is False for step in checkouts)


def test_host_tests_consume_the_shared_reference_vectors() -> None:
    vectors = json.loads(
        (ROOT / "tests" / "fixtures" / "parity" / "offset_contract.json").read_text(
            encoding="utf-8"
        )
    )
    assert vectors["offset_unit"] == "unicode_scalar"
    source = (CRATE / "tests" / "sidecar_qualification.rs").read_text(encoding="utf-8")
    assert "tests/fixtures/parity/offset_contract.json" in source
    lane = (CRATE / "tests" / "packaged_sidecar_lane.rs").read_text(encoding="utf-8")
    assert "tests/unit/service/fixtures/sidecar_stdio_golden.json" in lane
