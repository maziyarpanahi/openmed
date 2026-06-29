"""Adversarial evaluation attacks for benchmark reports."""

from .linkage import (
    LinkageAttackResult,
    LongitudinalLinkageAttackResult,
    linkage_attack,
    longitudinal_linkage_attack,
)
from .reid import (
    MembershipInferenceResult,
    ReidAttackResult,
    generate_reid_leaderboard,
    membership_inference_attack,
    render_reid_leaderboard,
    run_reid_attack,
    run_reid_benchmark,
)

__all__ = [
    "LinkageAttackResult",
    "LongitudinalLinkageAttackResult",
    "ReidAttackResult",
    "MembershipInferenceResult",
    "generate_reid_leaderboard",
    "linkage_attack",
    "longitudinal_linkage_attack",
    "membership_inference_attack",
    "render_reid_leaderboard",
    "run_reid_attack",
    "run_reid_benchmark",
]
