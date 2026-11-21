"""Rebuild the public synthetic pack from source and verify the committed bytes."""

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from openmed.eval.evidence_pack import ValidationEvidencePack
from openmed.eval.evidence_pack.build import build_evidence_pack
from openmed.eval.evidence_pack.verify import sign_evidence_pack, verify_evidence_pack

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "tests/fixtures/evidence_pack/deidentification_synthetic_v1"
GOLDEN = ROOT / "docs/benchmarks/golden.report.json"
TEST_KEY = b"public-synthetic-evidence-fixture-only"
CLOCK = datetime(2026, 6, 14, tzinfo=timezone.utc)


def rebuild(directory: Path) -> ValidationEvidencePack:
    """Use golden recall and explicitly synthetic controls for the other lanes."""
    golden = json.loads(GOLDEN.read_bytes())
    assert golden["metadata"]["synthetic"] is True and golden["fixture_count"] > 0
    control_path = FIXTURE / "control.json"
    control = json.loads(control_path.read_bytes())
    reference = {tuple(span) for span in control["reference"]}
    threshold = control["score_threshold"]
    active = {
        (a, b, label)
        for a, b, label, score in control["scored_spans"]
        if score >= threshold
    }
    quantized = {
        (a, b, label)
        for a, b, label, score in control["scored_spans"]
        if round(score * 127) / 127 >= threshold
    }
    direct_recall = len(reference & active) / len(reference)
    delta = abs(direct_recall - len(reference & quantized) / len(reference))
    leaked = sum(
        control["text"][a:b] in control["masked_text"] for a, b, _ in reference
    )
    invalid = sum(not 0 <= a < b <= len(control["text"]) for a, b, _ in active)
    values = [
        ("g1a", "direct_identifier_recall", direct_recall, direct_recall == 1),
        (
            "g2",
            "recall",
            golden["metrics"]["exact_span_f1"]["recall"],
            golden["metrics"]["exact_span_f1"]["false_negatives"] == 0,
        ),
        ("g3", "critical_leakage_count", leaked, leaked == 0),
        ("g4", "quantized_recall_delta", delta, delta == 0),
        ("span_integrity", "invalid_span_count", invalid, invalid == 0),
    ]
    index = []
    for lane, metric, value, passed in values:
        report = {
            "lane": lane,
            "state": "success" if passed else "failure",
            "metrics": {metric: value},
        }
        raw = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
        digest = hashlib.sha256(raw).hexdigest()
        (directory / f"{digest}.json").write_bytes(raw)
        index.append({"lane": lane, "type": "review_metrics", "sha256": digest})
    (directory / "reports.json").write_text(json.dumps(index))
    unsigned = build_evidence_pack(
        "deidentify", directory, version="4.3.0", generated_at=CLOCK
    ).to_dict()
    for name, path in [
        ("synthetic_golden_benchmark", GOLDEN),
        ("synthetic_scoring_control", control_path),
    ]:
        unsigned["frozen_inputs"].append(
            {
                "id": name,
                "origin": "synthetic",
                "digest": "sha256:"
                + hashlib.sha256(
                    path.read_text(encoding="utf-8").encode("utf-8")
                ).hexdigest(),
            }
        )
    unsigned["limitations"].append("synthetic_evidence_only")
    return sign_evidence_pack(
        ValidationEvidencePack(unsigned), key_id="fixture_only", key=TEST_KEY
    )


def test_rebuild_and_verify_committed_pack(tmp_path):
    actual = (FIXTURE / "pack.json").read_bytes()
    committed = ValidationEvidencePack.from_json(actual)
    rebuilt = rebuild(tmp_path)
    assert committed.digest() == rebuilt.digest()
    assert actual == rebuilt.to_json().encode()
    assert {
        "synthetic_evidence_only",
        "not_clinically_validated",
        "not_a_medical_device",
    } <= set(committed.to_dict()["limitations"])
    inputs = [
        row["id"]
        for row in committed.to_dict()["frozen_inputs"]
        if row["origin"] == "synthetic"
    ]
    assert inputs == ["synthetic_golden_benchmark", "synthetic_scoring_control"]
    params = dict(key_id="fixture_only", as_of=CLOCK, max_age=timedelta(days=1))
    assert verify_evidence_pack(actual, key=TEST_KEY, **params).exit_code == 0
    assert verify_evidence_pack(actual, key=b"x" * 32, **params).exit_code == 1
