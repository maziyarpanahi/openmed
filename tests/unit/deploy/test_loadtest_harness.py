"""Contract tests for the local k6 load-test harness."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCENARIO = ROOT / "deploy" / "loadtest" / "scenario.js"
RUNNER = ROOT / "deploy" / "loadtest" / "run.sh"
WORKFLOW = ROOT / ".github" / "workflows" / "loadtest.yml"
DOCS = ROOT / "docs" / "serving" / "load-testing.md"
MKDOCS = ROOT / "mkdocs.yml"


def test_scenario_covers_mixed_traffic_and_all_slo_metrics() -> None:
    content = SCENARIO.read_text(encoding="utf-8")

    assert '"/analyze"' in content
    assert '"/pii/deidentify"' in content
    assert '"/pii/extract/stream"' in content
    assert "p(95)<${" in content
    assert "p(99)<${" in content
    assert "rate<${" in content
    assert "rate>=${" in content
    assert "handleSummary" in content
    assert "LOADTEST_RESULT_FILE" in content


def test_runner_uses_an_ephemeral_loopback_container() -> None:
    content = RUNNER.read_text(encoding="utf-8")

    assert content.startswith("#!/usr/bin/env bash")
    assert "--rm" in content
    assert '--publish "127.0.0.1:${SERVICE_PORT}:8080"' in content
    assert "wait_for_ready" in content
    assert '"$DOCKER_BIN" rm --force' in content
    assert "synthetic_payload" in content


def test_nightly_workflow_archives_slo_results() -> None:
    content = WORKFLOW.read_text(encoding="utf-8")

    assert "schedule:" in content
    assert "workflow_dispatch:" in content
    assert "grafana/setup-k6-action@v1" in content
    assert "deploy/loadtest/run.sh" in content
    assert "actions/upload-artifact@v7" in content
    assert "if: always()" in content
    assert "loadtest-results/" in content


def test_load_testing_docs_are_in_navigation_and_describe_thresholds() -> None:
    docs = DOCS.read_text(encoding="utf-8")
    nav = MKDOCS.read_text(encoding="utf-8")

    assert "synthetic" in docs.lower()
    assert "LOADTEST_SLO_P95_MS" in docs
    assert "LOADTEST_SLO_P99_MS" in docs
    assert "LOADTEST_SLO_ERROR_RATE" in docs
    assert "LOADTEST_SLO_MIN_THROUGHPUT_RPS" in docs
    assert "serving/load-testing.md" in nav
