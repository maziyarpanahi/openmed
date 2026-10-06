"""Exercise each documented module entry point with networking disabled."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.integration
@pytest.mark.parametrize(
    "name",
    (
        "prior_auth_completeness",
        "chart_abstraction_evidence",
        "cohort_explanations",
        "quality_measure_evidence",
        "trial_eligibility_review",
    ),
)
def test_module_entry_points_finish_offline_with_stable_output(name):
    code = """
import runpy
import socket
import sys
def deny_network(*args, **kwargs):
    raise AssertionError('example attempted network access')
class OfflineSocket(socket.socket):
    def __new__(cls, *args, **kwargs):
        return deny_network(*args, **kwargs)
socket.socket = OfflineSocket
socket.create_connection = deny_network
socket.getaddrinfo = deny_network
runpy.run_module(sys.argv[1], run_name='__main__')
"""
    outputs = []
    for _ in range(2):
        completed = subprocess.run(
            [sys.executable, "-c", code, f"examples.{name}"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        assert completed.stderr == ""
        report = json.loads(completed.stdout)
        assert report["workflow_id"] == name
        assert report["passed"]
        assert report["fail_closed"]["code"]
        outputs.append(completed.stdout)
    assert outputs[0] == outputs[1]
