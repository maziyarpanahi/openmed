from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / ".gitleaks.toml"
BASELINE = ROOT / ".secrets.baseline"
CANARY = ROOT / "tests/fixtures/secret_scan_canary.txt"
PRE_COMMIT = ROOT / ".pre-commit-config.yaml"
WORKFLOW = ROOT / ".github/workflows/secret-scan.yml"


def _gitleaks_config_text() -> str:
    return CONFIG.read_text()


def test_pre_commit_runs_baseline_aware_gitleaks_hook() -> None:
    pre_commit = PRE_COMMIT.read_text()

    assert "https://github.com/gitleaks/gitleaks" in pre_commit
    assert "rev: v8.30.1" in pre_commit
    assert "- id: gitleaks" in pre_commit
    assert "--config=.gitleaks.toml" in pre_commit
    assert "--baseline-path=.secrets.baseline" in pre_commit


def test_canary_fixture_matches_project_specific_rule() -> None:
    config = _gitleaks_config_text()
    pattern_match = re.search(
        r"id = \"openmed-secret-scan-canary\".*?regex = '''(.+?)'''",
        config,
        flags=re.DOTALL,
    )

    assert pattern_match is not None
    assert re.search(pattern_match.group(1), CANARY.read_text()) is not None


def test_canary_fixture_is_the_only_canary_allowlist_path() -> None:
    config = _gitleaks_config_text()

    assert config.count('targetRules = ["openmed-secret-scan-canary"]') == 1
    assert "Committed canary fixture" in config
    assert r"tests/fixtures/secret_scan_canary\.txt$" in config


def test_canary_fixture_is_recorded_in_redacted_baseline() -> None:
    baseline = json.loads(BASELINE.read_text())

    assert [
        (
            finding["RuleID"],
            finding["File"],
            finding["Secret"],
            finding["Fingerprint"],
        )
        for finding in baseline
    ] == [
        (
            "openmed-secret-scan-canary",
            "tests/fixtures/secret_scan_canary.txt",
            "REDACTED",
            "tests/fixtures/secret_scan_canary.txt:openmed-secret-scan-canary:2",
        )
    ]


def test_local_credential_files_are_explicitly_flagged() -> None:
    config = _gitleaks_config_text()

    assert 'id = "local-credential-file"' in config
    assert r"creds\.txt" in config
    assert r"\.pypirc" in config
    assert r"secrets\.json" in config
    assert r"\.env" in config


def test_workflow_runs_canary_before_secret_scan() -> None:
    workflow = WORKFLOW.read_text()

    canary_step = workflow.index("Verify secret scanner canary")
    scan_step = workflow.index("Scan committed changes for secrets")
    assert canary_step < scan_step
    assert "secret_scan_canary.txt" in workflow
    assert "gitleaks git" in workflow
    assert "--baseline-path .secrets.baseline" in workflow
    assert "secret-scan" in workflow
