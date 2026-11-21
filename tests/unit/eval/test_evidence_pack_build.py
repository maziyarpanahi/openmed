import hashlib
import json
from datetime import datetime, timezone

import pytest

from openmed.cli.main import main
from openmed.eval.evidence_pack import EvidencePackError, ValidationEvidencePack
from openmed.eval.evidence_pack.build import build_evidence_pack
from openmed.eval.evidence_pack.lanes import required_lanes
from openmed.eval.report import BenchmarkReport

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def store(inputs, lane, kind, report):
    raw = json.dumps(report).encode()
    digest = hashlib.sha256(raw).hexdigest()
    (inputs / (digest + ".json")).write_bytes(raw)
    return {"lane": lane, "type": kind, "sha256": digest}


def test_cli_build_missing_and_atomic_overwrite(tmp_path, capsys):
    rows = []
    for lane, names in required_lanes("deidentify").items():
        report = BenchmarkReport(
            lane,
            "private-model",
            "cpu",
            1,
            dict.fromkeys(names, 0),
            metadata={"path": "/private/patient"},
        )
        rows.append(store(tmp_path, lane, "benchmark_report", report.to_dict()))
    (tmp_path / "reports.json").write_text(json.dumps(rows))
    (tmp_path / "private-source.json").write_text(
        "unreadable as JSON and never indexed"
    )
    output = tmp_path / "output.json"
    args = [
        "evidence",
        "build",
        "--capability",
        "deidentify",
        "--inputs",
        str(tmp_path),
        "--output",
        str(output),
    ]
    assert main(args) == 0
    original = output.read_bytes()
    pack = ValidationEvidencePack.from_json(original)
    assert not pack.lane_blockers()
    assert b"private" not in original
    (tmp_path / (rows[0]["sha256"] + ".json")).unlink()
    assert main(args) == 2
    assert output.read_bytes() == original
    assert main(args + ["--overwrite"]) == 1
    partial = ValidationEvidencePack.from_json(output.read_bytes())
    assert partial.to_dict()["signature"] is None
    assert partial.to_dict()["lanes"][0]["state"] == "unknown"
    assert "missing:g1a:report_digest" in partial.lane_blockers()
    assert "NOT_READY" in capsys.readouterr().out


def test_gate_and_review_projection(tmp_path):
    rows = [
        store(
            tmp_path,
            "g1a",
            "gate_report",
            {
                "gate_results": [
                    {
                        "gate": "G1a",
                        "passed": False,
                        "details": {"applicable_labels": ["NAME", "ID"]},
                    }
                ],
                "per_label_recall": {"NAME": 1, "ID": 0.75},
            },
        ),
        store(
            tmp_path,
            "span_integrity",
            "review_metrics",
            {
                "lane": "span_integrity",
                "state": "success",
                "metrics": {"invalid_span_count": 0, "private": 10},
            },
        ),
    ]
    (tmp_path / "reports.json").write_text(json.dumps(rows))
    pack = build_evidence_pack("deidentify", tmp_path, generated_at=NOW)
    lanes = {lane["id"]: lane for lane in pack.to_dict()["lanes"]}
    assert lanes["g1a"]["state"] == "failure"
    assert lanes["g1a"]["metrics"] == {"direct_identifier_recall": 0.75}
    assert lanes["span_integrity"]["metrics"] == {"invalid_span_count": 0}
    assert "private" not in pack.to_json()


def test_digest_mismatch_and_repeated_lane(tmp_path):
    row = store(tmp_path, "g3", "gate_report", {})
    (tmp_path / "reports.json").write_text(json.dumps([row]))
    (tmp_path / (row["sha256"] + ".json")).write_text("private-marker")
    with pytest.raises(EvidencePackError, match="^report_digest_mismatch$"):
        build_evidence_pack("deidentify", tmp_path, generated_at=NOW)
    (tmp_path / (row["sha256"] + ".json")).unlink()
    (tmp_path / "reports.json").write_text(json.dumps([row, row]))
    with pytest.raises(EvidencePackError, match="^invalid_report_index$"):
        build_evidence_pack("deidentify", tmp_path, generated_at=NOW)
