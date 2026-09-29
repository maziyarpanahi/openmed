"""Brief extension of the offline pipeline golden contract (synthetic only)."""

import json
import socket
import time
from pathlib import Path

from examples.v30_clinical_brief import run_example


def test_brief_pipeline_golden_is_offline_cited_and_bounded(monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("the synthetic pipeline attempted network access")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    started = time.monotonic()
    result = run_example()
    expected = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "fixtures/clinical/e2e/brief_golden.json"
        ).read_text()
    )
    assert result == expected
    assert time.monotonic() - started < 60
    brief = result["brief"]
    assert brief["status"] == "needs_review"
    assert len(brief["citations"]) == len(brief["verdicts"]) == 3
    assert result["grounding"][0]["candidate_count"] > 0
    assert "demo.patient@example.test" not in json.dumps(result)
    assert "212-555-0198" not in json.dumps(result)
    assert result["model_validation"] is False
