from __future__ import annotations

from dataclasses import replace

import pytest

from openmed.core.detector_plugins import DetectorCapability
from openmed.core.labels import CANONICAL_LABELS, PERSON
from openmed.core.policy import (
    PolicyCompilationError,
    PolicyVerificationError,
    compile_policy,
    list_policies,
    load_policy,
    policy_requirements,
    verify_policy_plan,
)


def _capability_without(*labels: str) -> DetectorCapability:
    covered = tuple(sorted(CANONICAL_LABELS - set(labels)))
    return DetectorCapability(
        name="synthetic_policy_detector",
        stage="fast_pii",
        covered_labels=covered,
    )


def test_all_bundled_policies_compile_to_verified_coverage_proofs():
    for policy_name in list_policies():
        compiled = compile_policy(policy_name)
        required_labels = {
            requirement.label for requirement in policy_requirements(policy_name)
        }

        assert compiled.policy_name == policy_name
        assert compiled.proof.verified is True
        assert compiled.proof.coverage_percent == 100.0
        assert {entry.label for entry in compiled.plan.entries} == required_labels
        assert {
            witness.label for witness in compiled.proof.witnesses
        } == required_labels

        checked = verify_policy_plan(
            load_policy(policy_name),
            compiled.plan,
        )
        assert checked.verified is True
        assert checked.plan_fingerprint == compiled.plan.fingerprint


def test_compiler_reports_precise_uncovered_label_for_synthetic_gap():
    profile = load_policy("hipaa_safe_harbor")

    with pytest.raises(PolicyCompilationError, match=PERSON) as exc_info:
        compile_policy(
            profile,
            detector_capabilities=(_capability_without(PERSON),),
        )

    assert exc_info.value.uncovered_labels == (PERSON,)
    assert "uncovered labels: PERSON" in str(exc_info.value)


def test_independent_checker_rejects_weaker_tampered_plan_action():
    profile = load_policy("hipaa_safe_harbor")
    capability = DetectorCapability(
        name="complete_policy_detector",
        stage="fast_pii",
        covered_labels="*",
    )
    compiled = compile_policy(profile, detector_capabilities=(capability,))
    weakened_entries = tuple(
        replace(entry, action="keep") if entry.label == PERSON else entry
        for entry in compiled.plan.entries
    )
    weakened_plan = replace(compiled.plan, entries=weakened_entries)

    with pytest.raises(PolicyVerificationError, match=PERSON) as exc_info:
        verify_policy_plan(
            profile,
            weakened_plan,
            detector_capabilities=(capability,),
        )

    assert exc_info.value.weak_actions[0].label == PERSON
    assert exc_info.value.weak_actions[0].observed_action == "keep"
    assert exc_info.value.proof is not None
    assert exc_info.value.proof.verified is False
