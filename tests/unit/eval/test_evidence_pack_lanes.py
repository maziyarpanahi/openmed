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
