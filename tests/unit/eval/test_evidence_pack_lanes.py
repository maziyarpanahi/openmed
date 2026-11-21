import pytest

from openmed.core.intended_use import RegulatoryTier, load_intended_use_registry
from openmed.eval.evidence_pack import EvidencePackError, ValidationEvidencePack
from openmed.eval.evidence_pack.lanes import required_lanes


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
