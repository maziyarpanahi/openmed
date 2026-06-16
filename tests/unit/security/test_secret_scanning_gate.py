from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / ".gitleaks.toml"
CANARY = ROOT / "tests/fixtures/secret_scan_canary.txt"
WORKFLOW = ROOT / ".github/workflows/ci.yml"


def _gitleaks_config() -> dict[str, object]:
    return tomllib.loads(CONFIG.read_text())


def test_canary_fixture_matches_project_specific_rule() -> None:
    config = _gitleaks_config()
    rules = {rule["id"]: rule for rule in config["rules"]}
    canary_rule = rules["openmed-secret-scan-canary"]

    assert re.search(canary_rule["regex"], CANARY.read_text()) is not None


def test_canary_fixture_is_the_only_canary_allowlist_path() -> None:
    config = _gitleaks_config()
    matching_allowlists = [
        allowlist
        for allowlist in config["allowlists"]
        if allowlist.get("targetRules") == ["openmed-secret-scan-canary"]
    ]

    assert matching_allowlists == [
        {
            "targetRules": ["openmed-secret-scan-canary"],
            "description": (
                "Committed canary fixture; CI copies it to a non-allowlisted path "
                "to prove detection still works"
            ),
            "paths": [r"tests/fixtures/secret_scan_canary\.txt$"],
        }
    ]


def test_local_credential_files_are_explicitly_flagged() -> None:
    config = _gitleaks_config()
    rules = {rule["id"]: rule for rule in config["rules"]}
    credential_rule = rules["local-credential-file"]

    assert "creds\\.txt" in credential_rule["path"]
    assert "\\.pypirc" in credential_rule["path"]
    assert "secrets\\.json" in credential_rule["path"]
    assert "\\.env" in credential_rule["path"]


def test_workflow_runs_canary_before_secret_scan() -> None:
    workflow = WORKFLOW.read_text()

    canary_step = workflow.index("Verify secret scanner canary")
    scan_step = workflow.index("Scan committed changes for secrets")
    assert canary_step < scan_step
    assert "secret_scan_canary.txt" in workflow
    assert "gitleaks git" in workflow
    assert "secret-scan" in workflow
