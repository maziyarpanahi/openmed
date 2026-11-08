"""Static policy checks for the packaged OpenMedKit Electron harness.

These run offline in the regular suite. The harness itself runs in the
dedicated ``Electron packaged harness`` workflow, which needs Electron and a
virtual display.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / "js" / "openmedkit-electron" / "harness"
WORKFLOW = ROOT / ".github" / "workflows" / "electron-harness.yml"
PINNED_ACTION = re.compile(r"uses:\s*[\w.-]+/[\w./-]+@[0-9a-f]{40}\s+#\s*v\d+")


def _workflow() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_harness_workflow_runs_packaged_app_headlessly_on_linux() -> None:
    workflow = _workflow()

    assert "runs-on: ubuntu-" in workflow
    assert "permissions:\n  contents: read" in workflow
    assert "run package" in workflow
    assert "xvfb-run" in workflow
    assert "npm --prefix js/openmedkit-electron/harness test" in workflow
    assert "npm --prefix js/openmedkit-electron test" in workflow
    assert "persist-credentials: false" in workflow
    uses = [line.strip() for line in workflow.splitlines() if "uses:" in line]
    assert uses
    assert all(PINNED_ACTION.search(line) for line in uses), uses


def test_harness_workflow_keeps_the_chromium_sandbox_enabled() -> None:
    workflow = _workflow()

    assert "--no-sandbox" not in workflow
    assert "ELECTRON_DISABLE_SANDBOX" not in workflow
    assert 'sudo chown root:root "$helper"' in workflow
    assert 'sudo chmod 4755 "$helper"' in workflow


def test_harness_workflow_is_the_only_one_driving_electron() -> None:
    workflows = sorted((ROOT / ".github" / "workflows").glob("*.yml"))
    drivers = [
        path.name
        for path in workflows
        if "js/openmedkit-electron/harness" in path.read_text(encoding="utf-8")
    ]

    assert drivers == ["electron-harness.yml"]


def test_packaging_is_local_unsigned_and_fuse_hardened() -> None:
    config = json.loads((HARNESS / "electron-builder.json").read_text("utf-8"))

    assert config["publish"] is None
    assert config["mac"]["identity"] is None
    assert config["mac"]["notarize"] is False
    assert config["win"]["signAndEditExecutable"] is False
    assert config["mac"]["target"] == ["dir"]
    assert config["linux"]["target"] == ["dir"]
    assert config["win"]["target"] == ["dir"]
    assert config["asar"] is True
    assert "afterSign" not in config
    assert config["electronDist"] == "node_modules/electron/dist"
    fuses = config["electronFuses"]
    assert fuses["runAsNode"] is False
    assert fuses["enableNodeOptionsEnvironmentVariable"] is False
    assert fuses["enableNodeCliInspectArguments"] is False
    assert fuses["onlyLoadAppFromAsar"] is True
    assert fuses["grantFileProtocolExtraPrivileges"] is False


def test_harness_toolchain_is_pinned_exactly() -> None:
    manifest = json.loads((HARNESS / "package.json").read_text("utf-8"))
    lock = json.loads((HARNESS / "package-lock.json").read_text("utf-8"))

    assert manifest["private"] is True
    for name, version in manifest["devDependencies"].items():
        assert re.fullmatch(r"\d+\.\d+\.\d+", version), (name, version)
        assert lock["packages"][f"node_modules/{name}"]["version"] == version
    assert "dependencies" not in manifest


def test_only_the_negative_control_relaxes_the_harness_sandbox() -> None:
    sources = {
        path.name: path.read_text(encoding="utf-8")
        for path in sorted((HARNESS / "src").glob("*.ts"))
    }
    relaxations = {name: text.count("sandbox: false") for name, text in sources.items()}

    assert relaxations == {name: 0 for name in sources} | {"main.ts": 1}
    assert "relax: { sandbox: false }" in sources["main.ts"]
    assert '"relaxed_window_refused"' in sources["main.ts"]
    for text in sources.values():
        assert "nodeIntegration: true" not in text
        assert "contextIsolation: false" not in text
        assert "webSecurity: false" not in text


def test_harness_uses_only_test_only_synthetic_bundles() -> None:
    stage = (HARNESS / "scripts" / "stage.mjs").read_text(encoding="utf-8")
    vectors = json.loads(
        (ROOT / "tests" / "fixtures" / "portable_bundle" / "vectors.json").read_text(
            "utf-8"
        )
    )

    assert vectors["synthetic_only"] is True
    assert vectors["test_only_keys"] is True
    assert 'tests", "fixtures", "portable_bundle", "vectors.json' in stage
    assert "synthetic_only !== true" in stage
