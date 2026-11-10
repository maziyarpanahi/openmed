"""Offline aECG intake, controlled reporting and review handoff."""

import json
from pathlib import Path

import pytest

from openmed.multimodal.aecg import AecgError, read_aecg


@pytest.mark.integration
def test_local_synthetic_aecg_review_handoff_without_network(monkeypatch):
    import socket

    monkeypatch.setattr(socket, "socket", lambda *a, **k: pytest.fail("network access"))
    fixture = Path(__file__).parents[1] / "fixtures/multimodal/aecg.xml"
    record = read_aecg(fixture.read_bytes())
    report = json.loads(json.dumps(record.report(), sort_keys=True))
    assert [series["kind"] for series in report["series"]] == [
        "rhythm",
        "derived_representative_beat",
    ]
    assert [series["sample_count"] for series in report["series"]] == [3, 2]
    assert report["review_required"]
    with pytest.raises(AecgError, match="review_required"):
        record.confirm_review(reviewer_confirmed=False)
    reviewed = record.confirm_review(reviewer_confirmed=True)
    assert reviewed.notice == report["notice"]
    assert reviewed.rhythm[0].leads[0].samples == (-4.0, 1.0, 11.0)
    assert "SYNTHETIC" not in json.dumps(report)
