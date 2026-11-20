"""Contract tests for the local k6 load-test harness."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from openmed.service.auth import authenticate_api_key, parse_api_key_credentials
from openmed.service.security_profile import run_service_preflight

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


@pytest.mark.skipif(os.name == "nt", reason="The container harness runs on Unix")
def test_runner_authenticates_production_warmup_and_load_clients(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run the shell wrapper and validate the emitted service/client contract."""
    if not shutil.which("bash") or not shutil.which("openssl"):
        pytest.skip("The harness requires Bash and OpenSSL")
    recorder = f"""#!{sys.executable}
import json
import os
import sys
from pathlib import Path
name = Path(sys.argv[0]).name
with open(os.environ['HARNESS_CALLS'], 'a') as output:
    output.write(json.dumps({{'tool': name, 'args': sys.argv[1:],
        'key': os.environ.get('LOADTEST_API_KEY') if name == 'k6' else None}}) + '\\n')
"""
    for name in ("docker", "curl", "k6"):
        executable = tmp_path / name
        executable.write_text(recorder, encoding="utf-8")
        executable.chmod(0o700)
    calls_path = tmp_path / "calls.jsonl"
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("HARNESS_CALLS", str(calls_path))
    monkeypatch.setenv("LOADTEST_SKIP_BUILD", "1")
    monkeypatch.setenv("LOADTEST_RESULT_FILE", str(tmp_path / "summary.json"))
    monkeypatch.setenv("OPENMED_PROFILE", "prod")
    result = subprocess.run(
        ["bash", str(RUNNER)], capture_output=True, text=True, check=True
    )
    calls = [json.loads(line) for line in calls_path.read_text().splitlines()]
    container = next(c for c in calls if c["tool"] == "docker")
    args = container["args"]
    config = dict(
        args[i + 1].split("=", 1) for i, arg in enumerate(args) if arg == "--env"
    )
    report = run_service_preflight(
        profile="prod", listen_host="0.0.0.0", listen_tls=False, environ=config
    )
    assert report.ready
    client_key = next(c["key"] for c in calls if c["tool"] == "k6")
    credentials = parse_api_key_credentials(config["OPENMED_SERVICE_AUTH_API_KEYS"])
    principal = authenticate_api_key(client_key, credentials)
    assert set(principal.scopes) == {"analyze:write", "pii:read", "pii:write"}
    warmup = [c for c in calls if c["tool"] == "curl" and "--data" in c["args"]]
    assert len(warmup) == 3
    assert all(f"X-API-Key: {client_key}" in c["args"] for c in warmup)
    assert client_key not in json.dumps(container)
    assert client_key not in result.stdout + result.stderr
    assert calls[-1]["tool"] == "docker"
    assert calls[-1]["args"][:2] == ["rm", "--force"]
    assert '"X-API-Key": __ENV.LOADTEST_API_KEY' in SCENARIO.read_text()


def test_nightly_workflow_archives_slo_results() -> None:
    content = WORKFLOW.read_text(encoding="utf-8")

    assert "schedule:" in content
    assert "workflow_dispatch:" in content
    assert "grafana/setup-k6-action@" in content
    assert "deploy/loadtest/run.sh" in content
    assert "actions/upload-artifact@" in content
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
