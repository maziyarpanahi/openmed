"""Run the service workflow under a real OS-enforced no-egress boundary.

The gate needs a Linux host that permits unprivileged user and network
namespaces (``unshare --user --map-root-user --net``), or a Docker daemon plus
a local image with the service extra named by ``OPENMED_NO_EGRESS_IMAGE``.
Hosts without either report a controlled skip; that skip is never a pass.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from openmed.core.attestation import (
    NO_EGRESS_NEGATIVE_CONTROLS,
    NO_EGRESS_POSITIVE_CONTROLS,
    NO_EGRESS_WORKFLOW_CASES,
    validate_no_egress_attestation,
)

ROOT = Path(__file__).resolve().parents[2]
GATE = ROOT / "scripts" / "security" / "no_egress_gate.py"
_TIMEOUT_SECONDS = 1200

pytestmark = pytest.mark.integration


def _run_gate(output: Path, generated_at: str) -> tuple[int, dict[str, Any], str]:
    completed = subprocess.run(
        [
            sys.executable,
            str(GATE),
            "--output",
            str(output),
            "--generated-at",
            generated_at,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=_TIMEOUT_SECONDS,
        check=False,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    return completed.returncode, payload, completed.stderr


@pytest.fixture(scope="module")
def gate_runs(tmp_path_factory: pytest.TempPathFactory) -> list[dict[str, Any]]:
    directory = tmp_path_factory.mktemp("no-egress")
    code, first, stderr = _run_gate(directory / "first.json", "2026-11-07T12:00:00Z")
    if code == 3:
        assert first["status"] == "skipped"
        image = os.environ.get("OPENMED_NO_EGRESS_IMAGE")
        pytest.skip(
            "no OS-enforced no-egress boundary on this host "
            f"(reason_code={first['reason_code']}, host={first['host']['kind']}); "
            "needs unprivileged user+network namespaces on Linux or Docker with "
            f"OPENMED_NO_EGRESS_IMAGE set (currently {'set' if image else 'unset'})"
        )
    assert code == 0, stderr
    code, second, stderr = _run_gate(directory / "second.json", "2026-11-08T08:30:00Z")
    assert code == 0, stderr
    return [first, second]


def test_cached_workflow_succeeds_with_only_loopback(gate_runs) -> None:
    payload = gate_runs[0]

    attestation = validate_no_egress_attestation(payload)
    assert attestation.passed
    assert payload["boundary"]["kind"] in {"netns", "container"}
    assert payload["boundary"]["isolated"] is True
    assert payload["boundary"]["interfaces"]["non_loopback_up"] == 0
    assert payload["counts"]["workflow"] == {
        "total": len(NO_EGRESS_WORKFLOW_CASES),
        "passed": len(NO_EGRESS_WORKFLOW_CASES),
        "failed": 0,
        "not_run": 0,
    }
    surfaces = {case["surface"] for case in payload["workflow"]}
    assert surfaces == {"rest", "graphql", "grpc", "journey", "lifecycle"}


def test_every_egress_negative_control_is_blocked(gate_runs) -> None:
    payload = gate_runs[0]

    assert [case["id"] for case in payload["negative_controls"]] == list(
        NO_EGRESS_NEGATIVE_CONTROLS
    )
    assert all(case["status"] == "blocked" for case in payload["negative_controls"])
    executors = {case["executor"] for case in payload["negative_controls"]}
    assert {"subprocess", "native_libc", "system_resolver"} <= executors
    assert [case["id"] for case in payload["positive_controls"]] == list(
        NO_EGRESS_POSITIVE_CONTROLS
    )
    assert all(case["status"] == "reachable" for case in payload["positive_controls"])


def test_cross_surface_results_agree(gate_runs) -> None:
    digests = {case["id"]: case["result_digest"] for case in gate_runs[0]["workflow"]}

    assert digests["rest_analyze"] == digests["graphql_analyze"]
    assert digests["rest_analyze"] == digests["grpc_analyze"]
    assert digests["rest_analyze"] == digests["service_restart"]
    assert digests["rest_extract"] == digests["grpc_extract"]
    assert digests["rest_deidentify"] == digests["graphql_deidentify"]


def test_attestation_is_deterministic_apart_from_declared_fields(gate_runs) -> None:
    first, second = (dict(run) for run in gate_runs)

    assert first["repro_hash"] == second["repro_hash"]
    assert first["generated_at"] != second["generated_at"]
    for field in first["volatile_fields"]:
        first.pop(field)
        second.pop(field)
    assert first == second


@pytest.mark.skipif(
    sys.platform == "win32", reason="the service lifecycle uses POSIX signals"
)
def test_workflow_cases_pass_on_loopback_without_the_os_boundary() -> None:
    """Exercise the workflow driver itself on any POSIX host, loopback only."""

    spec = importlib.util.spec_from_file_location("no_egress_gate_workflow", GATE)
    assert spec is not None and spec.loader is not None
    gate = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = gate
    try:
        spec.loader.exec_module(gate)
        fixture = gate.DEFAULT_FIXTURE
        results = gate.run_workflow(gate.load_fixture(fixture), fixture, verbose=False)
    finally:
        sys.modules.pop(spec.name, None)

    failed = {
        case["id"]: case["reason_code"]
        for case in results
        if case["status"] != "passed"
    }
    assert failed == {}
    assert [case["id"] for case in results] == list(NO_EGRESS_WORKFLOW_CASES)
