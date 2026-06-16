from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / ".gitleaks.toml"
CANARY = ROOT / "tests/fixtures/secret_scan_canary.txt"
WORKFLOW = ROOT / ".github/workflows/ci.yml"


def _gitleaks_config_text() -> str:
    return CONFIG.read_text()


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
    assert "secret-scan" in workflow
