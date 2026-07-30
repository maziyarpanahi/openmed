"""Adversarial evaluation attacks for benchmark reports."""

from openmed.eval.robustness import (
    AdversarialAttackArtifact,
    AdversarialRobustnessReport,
    ConfusableAttackCase,
    MixedScriptEvasionGateError,
    MixedScriptEvasionReport,
    adversarial_robustness_report,
    generate_confusable_attack_corpus,
    mixed_script_evasion_report,
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
    "ConfusableAttackCase",
    "LinkageAttackResult",
    "LongitudinalLinkageAttackResult",
    "MixedScriptEvasionGateError",
    "MixedScriptEvasionReport",
    "ReidAttackResult",
    "MembershipInferenceResult",
    "ShadowMembershipInferenceResult",
    "adversarial_robustness_report",
    "generate_reid_leaderboard",
    "generate_confusable_attack_corpus",
    "linkage_attack",
    "longitudinal_linkage_attack",
    "membership_inference_attack",
    "mixed_script_evasion_report",
    "replay_adversarial_attack",
    "render_reid_leaderboard",
    "run_reid_attack",
    "run_reid_benchmark",
    "shadow_membership_inference_attack",
    "unicode_defended_runner",
]
