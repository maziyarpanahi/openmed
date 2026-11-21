import hashlib
import json
from datetime import datetime, timezone

import pytest

from openmed.clinical.use_errors import UseErrorRecorder
from openmed.core.intended_use import RegulatoryTier, load_intended_use_registry
from openmed.core.schemas import schema_fingerprint
from openmed.eval.evidence_pack import EvidencePackError, ValidationEvidencePack
from openmed.eval.evidence_pack.build import build_evidence_pack
from openmed.eval.evidence_pack.lanes import required_lanes
from openmed.eval.reviewer_disagreement import (
    ReviewerDecision,
    reviewer_disagreement_report,
)


def test_all_declared_capabilities_have_schema_compatible_lanes():
    for code, declaration in load_intended_use_registry().declarations.items():
        profile = required_lanes(code)
        assert profile and all(profile.values())
        pack = ValidationEvidencePack(
            {
                "schema_version": "1.0.0",
                "capability": code,
                "version": "1.0.0",
                "generated_at": "2026-01-01T00:00:00Z",
                "intended_use_digest": declaration.digest(),
                "regulatory_tier": declaration.regulatory_tier.value,
                "lanes": [
                    {
                        "id": lane,
                        "state": "success",
                        "mandatory": True,
                        "report_digest": "sha256:" + "a" * 64,
                        "required_metrics": list(metrics),
                        "metrics": {metric: 0 for metric in metrics},
                        "claims": [],
                    }
                    for lane, metrics in profile.items()
                ],
                "frozen_inputs": [],
                "licenses": [],
                "exceptions": [],
                "claims": [],
                "limitations": ["not_clinically_validated", "not_a_medical_device"],
                "signature": None,
            }
        )
        assert not pack.lane_blockers()
        if declaration.regulatory_tier == RegulatoryTier.RESTRICTED:
            assert {"shadow", "human_factors"} <= profile.keys()


def test_privacy_profile_and_monotonic_tier_upgrades():
    base = required_lanes("deidentify")
    assert base == {
        "g1a": ("direct_identifier_recall",),
        "g2": ("recall",),
        "g3": ("critical_leakage_count",),
        "g4": ("quantized_recall_delta",),
        "span_integrity": ("invalid_span_count",),
    }
    previous = base
    for tier in (
        RegulatoryTier.INFORMATIONAL,
        RegulatoryTier.ASSISTIVE_REVIEW,
        RegulatoryTier.RESTRICTED,
    ):
        current = required_lanes("deidentify", tier=tier)
        assert previous.items() < current.items()
        previous = current
    assert required_lanes("ecg", tier="non_clinical") == required_lanes("ecg")
    base.clear()
    assert required_lanes("deidentify")


def test_unreviewed_inputs_fail_without_echoing_values():
    with pytest.raises(EvidencePackError, match="^unknown_capability$"):
        required_lanes("private")
    with pytest.raises(EvidencePackError, match="^unknown_regulatory_tier$"):
        required_lanes("deidentify", tier="private")


def human_factors_inputs():
    recorder = UseErrorRecorder()
    recorder.record("export_before_review", "decisions", "restricted")
    snapshot = recorder.snapshot()
    reviewers = [
        ReviewerDecision(f"private-case-{i}", reviewer, "yes")
        for i in range(5)
        for reviewer in ("private-alpha", "private-beta")
    ]
    return {
        "capability": "decisions",
        "use_error_snapshot": snapshot,
        "use_error_snapshot_digest": schema_fingerprint(snapshot),
        "reviewer_disagreement": reviewer_disagreement_report(reviewers).to_dict(),
    }


def build_human_factors_pack(path, data=None, kind="human_factors_report"):
    index = []
    if data is not None:
        raw = json.dumps(data).encode()
        digest = hashlib.sha256(raw).hexdigest()
        (path / (digest + ".json")).write_bytes(raw)
        index.append({"type": kind, "lane": "human_factors", "sha256": digest})
    (path / "reports.json").write_text(json.dumps(index))
    pack = build_evidence_pack(
        "decisions", path, generated_at=datetime.now(timezone.utc)
    )
    lane = next(x for x in pack.to_dict()["lanes"] if x["id"] == "human_factors")
    return pack, lane


def test_both_sources_are_bound_and_only_counts_escape(tmp_path):
    pack, lane = build_human_factors_pack(tmp_path, human_factors_inputs())
    assert lane["state"] == "success" and lane["mandatory"]
    assert lane["metrics"] == {"use_error_count": 1, "review_disagreement_rate": 0.0}
    assert {"use_error_snapshot", "reviewer_disagreement"} <= {
        r["id"] for r in pack.to_dict()["frozen_inputs"]
    }
    assert (
        "private-" not in pack.to_json()
        and "export_before_review" not in pack.to_json()
    )


def test_missing_inputs_unknown_and_generic_metrics_refused(tmp_path):
    for data in (None, {"capability": "decisions"}):
        pack, lane = build_human_factors_pack(tmp_path, data)
        assert "state:human_factors:unknown" in pack.lane_blockers()
        assert lane["metrics"] == {}
    with pytest.raises(EvidencePackError, match="human_factors_sources_required"):
        build_human_factors_pack(tmp_path, human_factors_inputs(), "review_metrics")
    data = human_factors_inputs()
    data["use_error_snapshot_digest"] = "sha256:" + "f" * 64
    with pytest.raises(EvidencePackError, match="use_error_digest_mismatch"):
        build_human_factors_pack(tmp_path, data)


def test_structured_membership_lane_uses_bounded_report(tmp_path):
    from openmed.eval.evidence_pack.lanes import membership_lane
    from openmed.risk.membership import membership_inference_self_test

    (tmp_path / "reports.json").write_text("[]")

    def build():
        return build_evidence_pack(
            "structured_release", tmp_path, generated_at=datetime.now(timezone.utc)
        )

    assert "state:membership_inference:unknown" in build().lane_blockers()
    rows = [{"group": "synthetic-private"}]
    source = membership_inference_self_test(
        rows, rows, quasi_identifiers=["group"], max_inference_rate=1
    ).to_dict()
    raw = json.dumps(source).encode()
    digest = hashlib.sha256(raw).hexdigest()
    (tmp_path / (digest + ".json")).write_bytes(raw)
    index = [
        {
            "lane": "membership_inference",
            "type": "membership_self_test",
            "sha256": digest,
        }
    ]
    (tmp_path / "reports.json").write_text(json.dumps(index))
    pack = build()
    assert not pack.lane_blockers()
    assert "synthetic-private" not in pack.to_json()
    assert membership_lane({**source, "max_inference_rate": None}) == ("unknown", {})
    assert (
        membership_lane({**source, "max_inference_rate": 0, "meets_policy": False})[0]
        == "failure"
    )
    with pytest.raises(EvidencePackError, match="invalid_membership_report"):
        membership_lane({**source, "unique_match_count": 0})
    index[0]["type"] = "review_metrics"
    (tmp_path / "reports.json").write_text(json.dumps(index))
    with pytest.raises(EvidencePackError, match="membership_self_test_required"):
        build()


def test_clinical_lane_native_reports_and_provenance(tmp_path):
    from datetime import timedelta

    from openmed.eval.evidence_pack.lanes import clinical_lane
    from openmed.eval.evidence_pack.verify import (
        sign_evidence_pack,
        verify_evidence_pack,
    )
    from openmed.eval.harness import run_context_multilingual_eval
    from openmed.eval.suites.sdoh_extraction import (
        SDOHBenchmarkUnavailable,
        SDOHUnavailableReason,
    )
    from openmed.eval.suites.temporal_consistency import run_temporal_consistency_suite
    from openmed.eval.suites.temporal_tlinks import evaluate_temporal_tlink_fixtures

    index = []

    def add(lane, kind, report):
        raw = json.dumps(report).encode()
        digest = hashlib.sha256(raw).hexdigest()
        (tmp_path / (digest + ".json")).write_bytes(raw)
        index.append({"lane": lane, "type": kind, "sha256": digest})

    add(
        "context_multilingual",
        "clinical_report",
        run_context_multilingual_eval().to_dict(),
    )
    add(
        "temporal_consistency",
        "clinical_report",
        run_temporal_consistency_suite().to_dict(),
    )
    for lane, names in required_lanes("clinical.assertion").items():
        if lane not in ("context_multilingual", "temporal_consistency"):
            add(
                lane,
                "review_metrics",
                {"lane": lane, "state": "success", "metrics": dict.fromkeys(names, 1)},
            )
    (tmp_path / "reports.json").write_text(json.dumps(index))
    now = datetime.now(timezone.utc)
    pack = build_evidence_pack("clinical.assertion", tmp_path, generated_at=now)

    def verify(pack):
        key = b"synthetic-clinical-verification-key"
        signed = sign_evidence_pack(pack, key_id="site", key=key)
        return verify_evidence_pack(
            signed.to_json(),
            key_id="site",
            key=key,
            as_of=now,
            max_age=timedelta(days=1),
        )

    assert not pack.to_dict()["claims"] and verify(pack).exit_code == 0
    forged = pack.to_dict()
    forged["claims"] = forged["lanes"][0]["claims"] = ["credentialed_corpus_measured"]
    assert (
        "credentialed_lane_required" in verify(ValidationEvidencePack(forged)).reasons
    )
    with pytest.raises(EvidencePackError, match="credentialed_provenance_required"):
        clinical_lane(
            "i2b2_2010_assertion",
            {"suite": "i2b2-2010-assertion", "metadata": {"data_origin": "synthetic"}},
            credentialed=True,
        )
    absent = SDOHBenchmarkUnavailable(SDOHUnavailableReason.NOT_CONFIGURED).to_dict()
    assert clinical_lane("shac_sdoh", absent, credentialed=True) == (
        "unknown",
        {},
        "shac_not_configured",
    )
    temporal = evaluate_temporal_tlink_fixtures().to_dict()
    assert (
        clinical_lane("temporal_tlinks", temporal, credentialed=False)[0] == "success"
    )
    temporal["gate"]["awareness"]["f1"] = 0.74
    assert (
        clinical_lane("temporal_tlinks", temporal, credentialed=False)[0] == "failure"
    )
