"""Offline DQD command-to-signed-bridge integration and negative controls."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from openmed.eval.suites.omop_quality import load_frozen_omop_quality_fixture
from openmed.interop.omop import (
    normalize_dqd_results_file,
    run_omop_quality_subprocess,
    verify_omop_quality_report,
)
from openmed.structured.store import StoreState

pytestmark = pytest.mark.integration
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "interop" / "omop"
DQD_FIXTURE = FIXTURES / "dqd_results_2_9_0.json"


def _request() -> dict[str, object]:
    fixture = load_frozen_omop_quality_fixture(FIXTURES / "quality_reconciliation.json")
    return {
        "artifact_type": "openmed.omop_quality_request",
        "schema_version": "1.0.0",
        "compatibility_policy": "same_major",
        "input": fixture.quality_input.to_dict(),
        "input_digest": fixture.quality_input.digest,
    }


def _run(path: Path, request: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "openmed.interop.omop.dqd", str(path)],
        input=json.dumps(request),
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )


@pytest.mark.parametrize("all_pass", [False, True])
def test_dqd_file_to_signed_report_runs_without_dqd_or_r(
    all_pass: bool, tmp_path: Path
) -> None:
    path = DQD_FIXTURE
    if all_pass:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload["CheckResults"]:
            row.update(
                failed=0, passed=1, isError=0, notApplicable=0, numViolatedRows=0
            )
        path = tmp_path / "all_pass.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
    fixture = load_frozen_omop_quality_fixture(FIXTURES / "quality_reconciliation.json")
    signing_key = b"synthetic-dqd-integration-key"
    result = run_omop_quality_subprocess(
        fixture.quality_input,
        command=(sys.executable, "-m", "openmed.interop.omop.dqd", str(path)),
        reconciliation=fixture.reconcile(),
        signing_key=signing_key,
        timeout=10,
    )
    assert result.state is (StoreState.SUCCESS if all_pass else StoreState.FAILURE)
    assert result.value is not None
    assert result.value.quality_verdict == ("pass" if all_pass else "fail")
    assert verify_omop_quality_report(result.value, signing_key)
    expected = normalize_dqd_results_file(path, quality_input=fixture.quality_input)
    assert result.value.source_output_digest == expected["output_digest"]
    assert "SENTINEL" not in result.value.to_json()
    command = _run(path, _request())
    assert command.returncode == 0
    assert command.stderr == ""
    assert json.loads(command.stdout) == expected


@pytest.mark.parametrize(
    "case,state",
    [
        ("version", "unsupported"),
        ("category", "unsupported"),
        ("malformed", "failure"),
        ("missing", "failure"),
        ("digest", "conflict"),
        ("request_version", "unsupported"),
        ("request_input", "failure"),
        ("request_shape", "failure"),
        ("request_size", "failure"),
    ],
)
def test_command_fails_closed_with_no_payload_or_private_path_in_diagnostics(
    case: str, state: str, tmp_path: Path
) -> None:
    payload = json.loads(DQD_FIXTURE.read_text(encoding="utf-8"))
    request = _request()
    path = tmp_path / "SYNTHETIC_PRIVATE_PATH.json"
    if case == "version":
        payload["Metadata"][0]["dqdVersion"] = "SYNTHETIC_VERSION_SENTINEL"
    elif case == "category":
        payload["CheckResults"][0]["category"] = "SYNTHETIC_CATEGORY_SENTINEL"
    elif case == "digest":
        request["input_digest"] = "sha256:" + "0" * 64
    elif case == "request_version":
        request["schema_version"] = "999.0.0"
    elif case == "request_input":
        request["input"] = "SYNTHETIC_INPUT_SENTINEL"
    elif case == "request_shape":
        request = {"privatePath": "SYNTHETIC_PATH_SENTINEL"}
    elif case == "request_size":
        request["input"] = "SYNTHETIC_INPUT_SENTINEL" * 10000
    if case != "missing":
        path.write_text(
            "SYNTHETIC_MALFORMED_SENTINEL"
            if case == "malformed"
            else json.dumps(payload),
            encoding="utf-8",
        )
    result = _run(path, request)
    assert result.returncode == 2
    assert result.stdout == ""
    diagnostic = json.loads(result.stderr)
    assert diagnostic["state"] == state
    assert set(diagnostic) == {"state", "code"}
    assert "SYNTHETIC" not in result.stderr
    assert "Traceback" not in result.stderr
    if case == "version":
        fixture = load_frozen_omop_quality_fixture(
            FIXTURES / "quality_reconciliation.json"
        )
        bridge = run_omop_quality_subprocess(
            fixture.quality_input,
            command=(sys.executable, "-m", "openmed.interop.omop.dqd", str(path)),
            reconciliation=fixture.reconcile(),
            signing_key=b"synthetic-dqd-key",
            timeout=10,
        )
        assert bridge.state is StoreState.FAILURE
        assert bridge.code == "quality_adapter_failed"
        assert bridge.value is None
