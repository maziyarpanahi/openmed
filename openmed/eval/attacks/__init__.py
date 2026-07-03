"""Adversarial evaluation attacks for benchmark reports."""

from openmed.eval.robustness import (
    AdversarialAttackArtifact,
    AdversarialRobustnessReport,
    adversarial_robustness_report,
    replay_adversarial_attack,
    unicode_defended_runner,
)

from .linkage import (
    LinkageAttackResult,
    LongitudinalLinkageAttackResult,
    linkage_attack,
    longitudinal_linkage_attack,
)
from .reid import (
    MembershipInferenceResult,
    ReidAttackResult,
    ShadowMembershipInferenceResult,
    generate_reid_leaderboard,
    membership_inference_attack,
    render_reid_leaderboard,
    run_reid_attack,
    run_reid_benchmark,
    shadow_membership_inference_attack,
)

__all__ = [
    "AdversarialAttackArtifact",
    "AdversarialRobustnessReport",
    "LinkageAttackResult",
    "LongitudinalLinkageAttackResult",
    "ReidAttackResult",
    "MembershipInferenceResult",
    "ShadowMembershipInferenceResult",
    "adversarial_robustness_report",
    "generate_reid_leaderboard",
    "linkage_attack",
    "longitudinal_linkage_attack",
    "membership_inference_attack",
    "replay_adversarial_attack",
    "render_reid_leaderboard",
    "run_reid_attack",
    "run_reid_benchmark",
    "shadow_membership_inference_attack",
    "unicode_defended_runner",
]
