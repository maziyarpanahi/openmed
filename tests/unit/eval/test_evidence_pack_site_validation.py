"""Verified site validation studies in ordinary evidence builds."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from openmed.clinical.validation import StudyConfig, run_validation_study
from openmed.eval.evidence_pack.build import build_evidence_pack
from openmed.eval.evidence_pack.lanes import site_validation_lane

KEY = b"site-reviewer-test-key-only"
SAMPLE = (
    Path(__file__).resolve().parents[3]
    / "tests/fixtures/clinical/validation_sample.jsonl"
)


def study(accepted=True):
    return run_validation_study(
        StudyConfig(
            dataset_path=SAMPLE,
            model_name="synthetic-model",
            dataset_id="/private/site/data",
            data_revision="site-revision-canary",
        ),
        runner=lambda fixture, *args: fixture.gold_spans if accepted else [],
        signing_key=KEY,
    ).to_dict()


def test_site_study_projected_in_pack(tmp_path):
    report = study()
    raw = json.dumps(report).encode()
    digest = hashlib.sha256(raw).hexdigest()
    (tmp_path / (digest + ".json")).write_bytes(raw)
    (tmp_path / "reports.json").write_text(
        json.dumps(
            [{"lane": "site_validation", "type": "validation_report", "sha256": digest}]
        )
    )
    pack = build_evidence_pack(
        "deidentify",
        tmp_path,
        generated_at=datetime.now(timezone.utc),
        site_validation_key=KEY,
    )
    lane = next(x for x in pack.to_dict()["lanes"] if x["id"] == "site_validation")
    assert lane["state"] == "success" and lane["report_digest"] == report["repro_hash"]
    assert lane["metrics"]["recall"] == 1
    assert {x["id"] for x in pack.to_dict()["frozen_inputs"]} >= {
        "site_validation_dataset_id",
        "site_validation_data_revision",
    }
    assert (
        "/private/site" not in pack.to_json()
        and "site-revision-canary" not in pack.to_json()
    )
    (tmp_path / (digest + ".json")).write_bytes(b"{}")
    pack = build_evidence_pack(
        "deidentify",
        tmp_path,
        generated_at=datetime.now(timezone.utc),
        site_validation_key=KEY,
    )
    lane = next(x for x in pack.to_dict()["lanes"] if x["id"] == "site_validation")
    assert lane["reason_code"] == "site_validation_unverified" and not lane["metrics"]


def test_unaccepted_and_untrusted_reports():
    report = study(False)
    lane, _ = site_validation_lane(report, KEY)
    assert (
        lane["state"] == "failure"
        and lane["reason_code"] == "site_validation_not_accepted"
    )
    for key in (None, b"other-key"):
        lane, refs = site_validation_lane(report, key)
        assert lane["state"] == "failure" and not refs and not lane["metrics"]
    report["metrics"]["overall"]["precision"] = 0.314
    lane, refs = site_validation_lane(report, KEY)
    assert lane["reason_code"] == "site_validation_unverified" and not refs
